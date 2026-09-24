# Выкладка инструмента разметки

**Где работает:** https://docsemenov.ru/razmetka/ (vps). Пароль — у капитана команды; в репозитории его нет.

| Что | Где на сервере |
|---|---|
| Код и снимки (только чтение) | `/opt/lct_labeler/app`, `/opt/lct_labeler/data/{png,images.json}` |
| Окружение Python | `/opt/lct_labeler/venv` (fastapi 0.115, uvicorn 0.34) |
| Настройки и пароль | `/etc/lct_labeler.env` (права 600, root) |
| Служба | `lct-labeler.service` — временный пользователь (DynamicUser), запись только в `/var/lib/lct_labeler`, слушает `127.0.0.1:8877` |
| **Разметка команды** | `/var/lib/lct_labeler/ann/<id>__<разметчик>.json` |
| nginx | `location ^~ /razmetka/` в `/etc/nginx/sites-available/docsemenov.ru`; копия до правки — `/root/docsemenov.nginx.bak_razmetka_*` |

```bash
# обновить код страницы/сервера
COPYFILE_DISABLE=1 tar --no-xattrs -C tools/labeler -czf - server.py static | ssh vps 'tar -xzf - -C /opt/lct_labeler/app && systemctl restart lct-labeler'
# забрать разметку к себе
ssh vps 'tar -C /var/lib/lct_labeler -czf - ann' | tar -xzf - -C data/work/labeler/
# состояние и журнал
ssh vps 'systemctl status lct-labeler --no-pager | head -5; journalctl -u lct-labeler -n 30 --no-pager'
# сменить имена в списке: LABEL_NAMES в /etc/lct_labeler.env → systemctl restart lct-labeler
```

**Снять после хакатона:** `systemctl disable --now lct-labeler`, удалить блок `/razmetka/` из nginx (`nginx -t && systemctl reload nginx`),
`rm -rf /opt/lct_labeler /etc/lct_labeler.env /etc/systemd/system/lct-labeler.service`; разметку из `/var/lib/lct_labeler` предварительно забрать.
