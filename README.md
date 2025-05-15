# 1. running backend
```bash
cd backend
chmod ./entrypoint.sh
./entrypoint.sh
```

# 2. running frontend
```bash 
npm install
npm run dev
```

# 3. docker + .env file
```bash
docker compose up --build
```
then go to `http://192.168.28.90:3000` for website, `:8000/docs` for fastAPI lab
