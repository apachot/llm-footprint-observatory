# Deploying the Web App on a Linux VPS

This web app is a plain Python HTTP server. The simplest production setup is:

- one Python virtual environment;
- one `systemd` service for `web/server.py`;
- one `nginx` reverse proxy in front;
- one `.env` file containing the OpenAI key.

## 1. Copy the project to the server

```bash
sudo mkdir -p /opt/ImpactLLM-parent
sudo chown "$USER":"$USER" /opt/ImpactLLM-parent
git clone https://github.com/apachot/ImpactLLM.git /opt/ImpactLLM-parent
cp -R /opt/ImpactLLM-parent/ImpactLLM /opt/ImpactLLM
cd /opt/ImpactLLM
```

## 2. Create the Python environment

```bash
cd /opt/ImpactLLM
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Configure OpenAI

Create `/opt/ImpactLLM/.env`:

```bash
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4.1-mini
```

## 4. Test locally on the server

```bash
cd /opt/ImpactLLM
LLM_WEB_HOST=127.0.0.1 LLM_WEB_PORT=8080 LLM_WEB_PREFIX=/impact-llm .venv/bin/python web/server.py
```

Then check:

```bash
curl http://127.0.0.1:8080/impact-llm/
```

## 5. Install the `systemd` service

Copy and adapt the service file:

```bash
sudo cp deploy/systemd/ImpactLLM-web.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ImpactLLM-web
sudo systemctl status ImpactLLM-web
```

If your Linux user is not `www-data`, change `User`, `Group`, `WorkingDirectory`, and `ExecStart` in the service file.

## 6. Install and configure `nginx`

Copy the reverse-proxy config:

```bash
sudo cp deploy/nginx/ImpactLLM.conf /etc/nginx/sites-available/ImpactLLM
sudo ln -s /etc/nginx/sites-available/ImpactLLM /etc/nginx/sites-enabled/ImpactLLM
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
journalctl -u ImpactLLM-web -f
```

## Notes

- The web app listens on `LLM_WEB_HOST` and `LLM_WEB_PORT`.
- Set `LLM_WEB_PREFIX=/impact-llm` when the app is exposed under `https://dev.emotia.com/impact-llm`.
- Keep the Python server bound to `127.0.0.1` and expose it through `nginx`.
- This app uses the Python standard library HTTP server. For moderate traffic and a research/demo site, that is usually sufficient. For heavier traffic, put it behind a stronger application server or rewrite it on a production WSGI/ASGI stack.
