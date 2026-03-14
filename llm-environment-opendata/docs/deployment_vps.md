# Deploying the Web App on a Linux VPS

This web app is a plain Python HTTP server. The simplest production setup is:

- one Python virtual environment;
- one `systemd` service for `web/server.py`;
- one `nginx` reverse proxy in front;
- one `.env` file containing the OpenAI key.

## 1. Copy the project to the server

```bash
sudo mkdir -p /opt/llm-environment-opendata-parent
sudo chown "$USER":"$USER" /opt/llm-environment-opendata-parent
git clone https://github.com/apachot/ImpactLLM.git /opt/llm-environment-opendata-parent
cp -R /opt/llm-environment-opendata-parent/llm-environment-opendata /opt/llm-environment-opendata
cd /opt/llm-environment-opendata
```

## 2. Create the Python environment

```bash
cd /opt/llm-environment-opendata
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Configure OpenAI

Create `/opt/llm-environment-opendata/.env`:

```bash
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4.1-mini
```

## 4. Test locally on the server

```bash
cd /opt/llm-environment-opendata
LLM_WEB_HOST=127.0.0.1 LLM_WEB_PORT=8080 LLM_WEB_PREFIX=/impact-llm .venv/bin/python web/server.py
```

Then check:

```bash
curl http://127.0.0.1:8080/impact-llm/
```

## 5. Install the `systemd` service

Copy and adapt the service file:

```bash
sudo cp deploy/systemd/llm-environment-opendata-web.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now llm-environment-opendata-web
sudo systemctl status llm-environment-opendata-web
```

If your Linux user is not `www-data`, change `User`, `Group`, `WorkingDirectory`, and `ExecStart` in the service file.

## 6. Install and configure `nginx`

Copy the reverse-proxy config:

```bash
sudo cp deploy/nginx/llm-environment-opendata.conf /etc/nginx/sites-available/llm-environment-opendata
sudo ln -s /etc/nginx/sites-available/llm-environment-opendata /etc/nginx/sites-enabled/llm-environment-opendata
sudo nginx -t
sudo systemctl reload nginx
```

## 7. Add HTTPS with Let's Encrypt

If the server has a real domain:

```bash
sudo apt-get update
sudo apt-get install -y certbot python3-certbot-nginx
sudo certbot --nginx -d dev.emotia.com
```

## 8. Logs

```bash
journalctl -u llm-environment-opendata-web -f
```

## Notes

- The web app listens on `LLM_WEB_HOST` and `LLM_WEB_PORT`.
- Set `LLM_WEB_PREFIX=/impact-llm` when the app is exposed under `https://dev.emotia.com/impact-llm`.
- Keep the Python server bound to `127.0.0.1` and expose it through `nginx`.
- This app uses the Python standard library HTTP server. For moderate traffic and a research/demo site, that is usually sufficient. For heavier traffic, put it behind a stronger application server or rewrite it on a production WSGI/ASGI stack.
