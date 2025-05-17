# Lab02: Serving model qua API và đóng gói với docker

# Members
| Name                | MSSV      | Roles  |
|---------------------|-----------|--------|
| Nguyễn Hữu Nam      | 22520917  | Leader |
| Nguyễn Khánh        | 22520641  | Member |
| Nguyễn Minh Sơn        | 22521254  | Member |

# Structure
```bash
├── README.md
├── backend
│   ├── Dockerfile
│   └── entrypoint_local.sh
├── docker-compose.yml
├── frontend
│   ├── Dockerfile
│   └──.env
└── .env
```
## Intructions
1. Tạo file .env ở thư mục root với nội dung (link [key](https://drive.google.com/file/d/12J9SV2gm4PFg_6vuTpBw8j0bKCENeWzq/view?usp=sharing)):
```bash
WANDB_API_KEY= # for security reason there is no api key here
```
2. Tạo file .env ở thư mục `frontend` với nội dung:
```bash
VITE_BACKEND_API=http://192.168.28.90:8000
```
3. Chạy server với docker compose:
```bash
docker compose up --build
```
4. Truy cập `http://192.168.28.90:3000` để thử nghiệm

## Videos
- links: https://drive.google.com/file/d/1--ARQtPlUUUzdJU4m6sgbSpK51RD3xdH/view?usp=drive_link

## Demo locally (no docker, for dev stuff)
# 1. running backend
```bash
cd backend
chmod ./entrypoint_local.sh
./entrypoint.sh
```

# 2. running frontend
```bash 
npm install
npm run dev
```