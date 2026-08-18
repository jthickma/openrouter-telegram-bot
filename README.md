# OpenRouter Telegram Bot

A self-hosted Telegram bot where every Telegram user authenticates with their own OpenRouter API key, browses OpenRouter's current model catalog, chooses a model, and sends text, images, PDFs, text/code files, or model-native files from the chat.

The bot does not require a shared inference key. User keys are validated against OpenRouter, held only in process memory, and never written to the usage or settings files.

## Features

- Live chat-model selection from OpenRouter's [`GET /api/v1/models`](https://openrouter.ai/docs/api/api-reference/models/get-models) endpoint.
- Live image-model selection and generation through OpenRouter's [Image API](https://openrouter.ai/docs/guides/overview/multimodal/image-generation).
- Text inference through [`POST /api/v1/chat/completions`](https://openrouter.ai/docs/api/api-reference/chat/create-a-chat-completion).
- Image understanding with Telegram photos and image documents encoded as private base64 data URLs, following OpenRouter's [image-input format](https://openrouter.ai/docs/guides/overview/multimodal/image-understanding).
- PDF inference with OpenRouter's `file` content type and configurable [PDF parsing engine](https://openrouter.ai/docs/guides/overview/multimodal/pdfs).
- UTF-8 text, source code, JSON, CSV, Markdown, XML, YAML, and similar files embedded as text for any text model.
- Other file types sent to models that advertise native `file` input support.
- Streaming Telegram responses.
- Exact per-request cost accounting from OpenRouter's returned `usage.cost`, not a static token-price estimate.
- Per-user local soft budgets plus deployment-wide budgets and OpenRouter key-limit reporting.
- Access controls for private and group chats.
- Docker deployment as an unprivileged user with persistent non-secret settings and usage volumes.

## How the OpenRouter integration works

| Bot action | OpenRouter behavior |
|---|---|
| `/key` | Validates the bearer token with [`GET /api/v1/key`](https://openrouter.ai/docs/api/api-reference/api-keys/get-current-api-key). |
| `/models`, `/model` | Fetches current text-output models and their input modalities, context size, supported parameters, and pricing from the [Models API](https://openrouter.ai/docs/guides/overview/models). |
| Text message | Sends the selected model and conversation to `/api/v1/chat/completions`. Only optional parameters advertised in `supported_parameters` are included. |
| Telegram photo | Sends text first, then an `image_url` base64 data URL. The selected model must advertise `image` input. |
| PDF | Sends a `file` content part. PDFs work with any OpenRouter text model because OpenRouter can parse them before inference. |
| Text/code file | Decodes UTF-8 locally and sends it as text, allowing any text model to analyze it. |
| Other file | Sends a `file` content part only when the selected model advertises native `file` input. |
| `/imagemodels`, `/image` | Discovers image-output models and sends generation requests to `POST /api/v1/images`. |
| `/stats`, budgets | Records the native token counts and actual cost in the final normal response or final streaming SSE event, as documented in [Usage Accounting](https://openrouter.ai/docs/cookbook/administration/usage-accounting). |

The bot also supplies a stable pseudonymous `user` value on chat requests. OpenRouter documents this field as an end-user identifier for abuse isolation and says it is hashed before being sent upstream.

## Quick start with Docker Compose

Prerequisites:

- A Telegram bot token from [BotFather](https://core.telegram.org/bots/tutorial#obtain-your-bot-token).
- Docker with Compose.
- Each user needs an [OpenRouter API key](https://openrouter.ai/settings/keys) with credits or access to free models.

Create the environment file:

```sh
cp .env.example .env
```

Set at least:

```dotenv
TELEGRAM_BOT_TOKEN=123456:replace-with-botfather-token
ADMIN_USER_IDS=-
ALLOWED_TELEGRAM_USER_IDS=*
```

Build and run:

```sh
docker compose up --build -d
docker compose logs -f openrouter-telegram-bot
```

Then open a private chat with the bot:

```text
/key sk-or-v1-your-key
/models image
/model google/gemini-2.5-flash
Describe what makes a rainbow.
```

Use the exact model slug shown by `/models`; model availability changes over time.

## Commands

| Command | Description |
|---|---|
| `/key OPENROUTER_KEY` | Validate and keep your key in memory. Private chats only. |
| `/keyinfo` | Refresh current OpenRouter key usage, limit, remaining limit, and expiry. |
| `/logout` | Forget the in-memory key and clear the user's in-memory conversations. |
| `/models [input] [search]` | Browse current text-output models. Optional input is `text`, `image`, `file`, `audio`, or `video`. |
| `/model [provider/model]` | Show or select the current chat model. Model changes reset conversation history. |
| `/imagemodels [search]` | Browse current image-output models. |
| `/imagemodel [provider/model]` | Show or select the image-generation model. |
| `/image PROMPT` | Generate one image using the selected image model. |
| `/budget` | Show the local soft cap and deployment-wide remaining budget. |
| `/budget AMOUNT PERIOD` | Set a soft cap. Period: `daily`, `weekly`, `monthly`, or `all-time`. |
| `/budget off` | Disable the user's local soft cap. |
| `/stats` | Show exact local costs, tokens, per-model totals, and refreshed OpenRouter key usage. |
| `/reset` | Reset the current user/chat/topic conversation. |
| `/resend` | Repeat the last text prompt in the current conversation. |
| `/chat PROMPT` | Ask the bot in a group chat. |

Model browser buttons use the latest catalog fetched for that user. A direct `/model provider/model` selection is also validated against the current API catalog.

## Files and multimodal input

### Images

Supported OpenRouter image-input media types are PNG, JPEG, WebP, and GIF. Telegram photos arrive as JPEG; image documents retain their declared MIME type. Choose a model returned by `/models image` before uploading.

The request follows OpenRouter's recommendation to put the text prompt before the image part. A Telegram caption becomes the prompt; otherwise the bot asks the model to describe and analyze the image.

### PDFs

PDFs are sent as base64 `data:application/pdf` file parts. The default parser is `cloudflare-ai`, which OpenRouter documents as a free PDF-to-Markdown engine. Set `OPENROUTER_PDF_ENGINE=mistral-ocr` for scanned/image-heavy PDFs or `native` to require native provider handling.

`mistral-ocr` can add per-page charges. OpenRouter's PDF documentation should be reviewed before enabling it. File annotations returned by OpenRouter are retained in conversation history while the large original file data is removed, allowing follow-up questions without keeping the full uploaded binary in bot memory.

### Text and code files

Known text MIME types and common code/data extensions are decoded as UTF-8 and wrapped with filename boundaries in the prompt. `TEXT_FILE_MAX_CHARS` limits the decoded content before it is sent.

### Other files

Non-PDF binary files are accepted only if the chosen model advertises `file` in `architecture.input_modalities`. Use `/models file` to browse them. Provider support still varies; an OpenRouter/provider error is returned to Telegram if the specific file format is rejected.

`MAX_FILE_SIZE_MB` limits every downloaded Telegram attachment before base64 encoding. Remember that base64 increases the API request size by roughly one third.

## Budgets and cost accounting

OpenRouter pricing varies by model and may include input/output tokens, images, requests, reasoning, caching, or PDF parsing. For that reason this bot uses the `usage.cost` value returned by OpenRouter as the authoritative local cost.

There are three layers:

1. **OpenRouter key limit:** a hard server-side control configured when creating/managing the user's OpenRouter key. `/keyinfo` and `/stats` show `limit_remaining` when available.
2. **User soft budget:** set from Telegram with `/budget 5 monthly`. The bot checks recorded spend before a request.
3. **Deployment budget:** the operator can set `USER_BUDGETS`, `GUEST_BUDGET`, and `BUDGET_PERIOD`.

Bot-side budgets are pre-request soft caps. Because the final cost is known only after inference, one request can cross the cap. Use a restricted OpenRouter API key or OpenRouter [guardrail budget](https://openrouter.ai/docs/guides/features/guardrails/overview) when a hard server-side limit is required.

## Security and privacy

- Do not set a shared `OPENROUTER_API_KEY`; this bot intentionally uses each Telegram user's key.
- `/key` is refused in groups. The bot immediately attempts to delete the private Telegram message containing the key.
- Telegram bot chats are not end-to-end encrypted. Message deletion is best effort and does not erase Telegram's infrastructure copies. Create a dedicated OpenRouter key with the smallest practical spending limit and revoke/rotate it if exposed.
- Keys exist only in process memory. They are not stored in `user_data/settings.json`, `usage_logs`, logs, Docker volumes, or command lines. Users must authenticate again after a restart or redeployment.
- `/logout` removes the key and the user's in-memory conversation history.
- Non-secret model preferences and local-budget settings persist in `user_data`; exact daily costs and tokens persist in `usage_logs`.
- Prompts and attachments are sent to OpenRouter and the selected inference provider. Review OpenRouter's [data collection](https://openrouter.ai/docs/guides/privacy/data-collection) and provider-logging documentation for the account's privacy configuration.
- Set `ALLOWED_TELEGRAM_USER_IDS` for private deployments. An unrestricted public bot can be abused even though users supply their own inference keys.

## Configuration

### Required

| Variable | Default | Description |
|---|---:|---|
| `TELEGRAM_BOT_TOKEN` | none | Telegram bot token from BotFather. |

### OpenRouter

| Variable | Default | Description |
|---|---:|---|
| `OPENROUTER_DEFAULT_MODEL` | `openrouter/auto` | Initial chat model before a user selects another model. |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | API base URL. Keep the default for OpenRouter. |
| `OPENROUTER_HTTP_REFERER` | empty | Optional app-attribution URL sent as `HTTP-Referer`. |
| `OPENROUTER_APP_TITLE` | `OpenRouter Telegram Bot` | Optional app-attribution title sent as `X-Title`. |
| `OPENROUTER_PDF_ENGINE` | `cloudflare-ai` | `cloudflare-ai`, `mistral-ocr`, or `native`. |
| `OPENROUTER_REQUEST_TIMEOUT` | `180` | HTTP request timeout in seconds. |
| `OPENROUTER_PROXY` | empty | Proxy used only for OpenRouter. `PROXY` overrides it. |

OpenRouter documents `HTTP-Referer` and `X-Title` as optional headers in its [quickstart](https://openrouter.ai/docs/quickstart). Do not put secrets in either value.

### Access and deployment budgets

| Variable | Default | Description |
|---|---:|---|
| `ADMIN_USER_IDS` | `-` | Comma-separated Telegram admins; admins bypass deployment budgets. |
| `ALLOWED_TELEGRAM_USER_IDS` | `*` | Comma-separated user IDs or `*`. |
| `BUDGET_PERIOD` | `monthly` | `daily`, `monthly`, or `all-time`. |
| `USER_BUDGETS` | `*` | Comma-separated USD caps aligned with allowed IDs, or `*` for unlimited. |
| `GUEST_BUDGET` | `100` | Shared USD cap for group users outside the allowed list. |

### Inference and Telegram

| Variable | Default | Description |
|---|---:|---|
| `ASSISTANT_PROMPT` | `You are a helpful assistant.` | System prompt for new conversations. |
| `STREAM` | `true` | Stream text completions into edited Telegram messages. |
| `SHOW_USAGE` | `true` | Add the actual model, token count, and cost after a response. |
| `MAX_TOKENS` | `4096` | Included only when the selected model advertises `max_tokens`. |
| `TEMPERATURE` | `0.7` | Included only for models that advertise it. |
| `PRESENCE_PENALTY` | `0` | Included only for models that advertise it. |
| `FREQUENCY_PENALTY` | `0` | Included only for models that advertise it. |
| `MAX_HISTORY_SIZE` | `15` | Maximum in-memory messages per user/chat/topic, including the system prompt. |
| `MAX_CONVERSATION_AGE_MINUTES` | `180` | Reset idle conversation history after this duration. |
| `ENABLE_IMAGE_GENERATION` | `true` | Enable `/image`. |
| `ENABLE_QUOTING` | `true` | Reply to the originating Telegram message. |
| `GROUP_TRIGGER_KEYWORD` | empty | Optional prefix required for non-`/chat` group prompts. |
| `IGNORE_GROUP_ATTACHMENTS` | `true` | Ignore images/files in groups unless explicitly disabled. |
| `MAX_FILE_SIZE_MB` | `10` | Maximum attachment size downloaded and sent to OpenRouter. |
| `TEXT_FILE_MAX_CHARS` | `200000` | Maximum decoded UTF-8 characters per text/code file. |
| `PROXY` | empty | Shared Telegram and OpenRouter proxy. |
| `TELEGRAM_PROXY` | empty | Telegram-only proxy when `PROXY` is unset. |
| `USER_SETTINGS_PATH` | `user_data/settings.json` | Non-secret preferences file. |
| `USAGE_LOGS_DIR` | `usage_logs` | Exact usage/cost JSON directory. |

## Run from source

Python 3.11 or newer is required; the Docker image uses Python 3.12.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
python bot/main.py
```

For development and tests:

```sh
python -m pip install -r requirements-dev.txt
pytest
```

## Deployment verification

Before treating a deployment as complete, verify on the actual host:

1. `docker compose config` resolves the intended `.env` and persistent volumes.
2. The bot starts without exposing the Telegram token in logs.
3. A limited test key authenticates through `/key` and its Telegram message is deleted.
4. `/models image` returns live models and button selection updates `/model`.
5. One text request, one image-understanding request, one text file, and one PDF succeed with the selected model.
6. `/stats` cost matches the OpenRouter Activity page for those requests.
7. A deliberately tiny local budget blocks the next request, and the OpenRouter key's hard limit is also configured as intended.
8. Restarting the container requires `/key` again while model preferences and usage totals remain.

Local unit tests can validate request construction and accounting, but they cannot prove Telegram delivery, provider-specific file acceptance, a real model response, or host filesystem permissions without live credentials.

## Official OpenRouter references

- [Quickstart and authentication](https://openrouter.ai/docs/quickstart)
- [Models API reference](https://openrouter.ai/docs/api/api-reference/models/get-models)
- [Model capability fields and supported parameters](https://openrouter.ai/docs/guides/overview/models)
- [Chat Completions API](https://openrouter.ai/docs/api/api-reference/chat/create-a-chat-completion)
- [Multimodal overview](https://openrouter.ai/docs/guides/overview/multimodal/overview)
- [Image input](https://openrouter.ai/docs/guides/overview/multimodal/image-understanding)
- [PDF input and parsing](https://openrouter.ai/docs/guides/overview/multimodal/pdfs)
- [Image generation](https://openrouter.ai/docs/guides/overview/multimodal/image-generation)
- [Usage accounting](https://openrouter.ai/docs/cookbook/administration/usage-accounting)
- [Current API key details](https://openrouter.ai/docs/api/api-reference/api-keys/get-current-api-key)
- [Generation metadata](https://openrouter.ai/docs/api/api-reference/generations/get-generation)
- [Privacy and data collection](https://openrouter.ai/docs/guides/privacy/data-collection)

## License

GPL-2.0. See [LICENSE](LICENSE).
