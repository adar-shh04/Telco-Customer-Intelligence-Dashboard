# Docker Deployment Guide - Streamlit Dashboard

Complete step-by-step guide to deploy the Telco Customer Intelligence dashboard using Docker.

---

## 📋 Prerequisites

Before starting, ensure you have:

1. **Docker Desktop** installed on Windows
   - Download from: https://www.docker.com/products/docker-desktop
   - Install and restart your computer if needed
   - Verify installation: `docker --version`

2. **Project files** ready
   - All data files in `data/raw/` directory
   - Streamlit dashboard in `dashboard/` directory
   - `requirements.txt` file present

---

## 🚀 Step-by-Step Deployment

### Step 1: Verify Docker Installation

Open PowerShell or Command Prompt and run:

```powershell
docker --version
```

**Expected output:**
```
Docker version 24.x.x, build xxxxxxx
```

If you see an error, Docker is not installed or not running.

---

### Step 2: Navigate to Project Directory

```powershell
cd "path/to/your/Telco Customer Intelligence Platform"
```

---

### Step 3: Build Docker Image

Build the Docker image with a tag name:

```powershell
docker build -t telecom-dashboard:latest .
```

**What this does:**
- `-t telecom-dashboard:latest` - Tags the image with name "telecom-dashboard" and version "latest"
- `.` - Uses the current directory (where Dockerfile is located)

**Expected output:**
```
[+] Building 45.2s (10/10) FINISHED
 => [1/5] FROM docker.io/library/python:3.9-slim
 => [2/5] WORKDIR /app
 => [3/5] COPY requirements.txt .
 => [4/5] RUN pip install --no-cache-dir -r requirements.txt
 => [5/5] COPY . .
 => exporting to image
 => => naming to docker.io/library/telecom-dashboard:latest
```

**Time:** This will take 2-5 minutes depending on your internet speed.

---

### Step 4: Verify Image Was Created

Check if the image was built successfully:

```powershell
docker images
```

**Expected output:**
```
REPOSITORY           TAG       IMAGE ID       CREATED          SIZE
telecom-dashboard    latest    abc123def456   2 minutes ago    1.2GB
```

---

### Step 5: Run Docker Container

Run the container and map ports:

```powershell
docker run -d -p 8501:8501 --name telecom-app telecom-dashboard:latest
```

**What this does:**
- `-d` - Run in detached mode (background)
- `-p 8501:8501` - Map port 8501 from container to your local machine
- `--name telecom-app` - Give the container a friendly name
- `telecom-dashboard:latest` - Use the image we just built

**Expected output:**
```
a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1v2w3x4y5z6
```
(This is the container ID)

---

### Step 6: Verify Container is Running

Check if the container is running:

```powershell
docker ps
```

**Expected output:**
```
CONTAINER ID   IMAGE                        COMMAND                  STATUS         PORTS                    NAMES
abc123def456   telecom-dashboard:latest     "streamlit run dash…"    Up 10 seconds  0.0.0.0:8501->8501/tcp   telecom-app
```

---

### Step 7: Access Your Dashboard

Open your web browser and navigate to:

```
http://localhost:8501
```

**You should see your Streamlit dashboard running!** 🎉

---

## 🔧 Common Docker Commands

### View Running Containers
```powershell
docker ps
```

### View All Containers (including stopped)
```powershell
docker ps -a
```

### View Container Logs
```powershell
docker logs telecom-app
```

### View Live Logs (follow mode)
```powershell
docker logs -f telecom-app
```

### Stop Container
```powershell
docker stop telecom-app
```

### Start Stopped Container
```powershell
docker start telecom-app
```

### Restart Container
```powershell
docker restart telecom-app
```

### Remove Container
```powershell
docker stop telecom-app
docker rm telecom-app
```

### Remove Image
```powershell
docker rmi telecom-dashboard:latest
```

### Execute Command Inside Container
```powershell
docker exec -it telecom-app bash
```

---

## 🐛 Troubleshooting

### Issue 1: Port Already in Use

**Error:**
```
Error: bind: address already in use
```

**Solution:**
Stop any process using port 8501:

```powershell
# Find process using port 8501
netstat -ano | findstr :8501

# Kill the process (replace PID with actual process ID)
taskkill /PID <PID> /F

# Or use a different port
docker run -d -p 8502:8501 --name telecom-app telecom-dashboard:latest
# Then access at http://localhost:8502
```

---

### Issue 2: Container Exits Immediately

**Check logs:**
```powershell
docker logs telecom-app
```

**Common causes:**
- Missing data files
- Python dependencies not installed
- Syntax errors in code

**Solution:**
Fix the issue and rebuild:
```powershell
docker stop telecom-app
docker rm telecom-app
docker build -t telecom-dashboard:latest .
docker run -d -p 8501:8501 --name telecom-app telecom-dashboard:latest
```

---

### Issue 3: Cannot Access Dashboard

**Check if container is running:**
```powershell
docker ps
```

**Check container logs:**
```powershell
docker logs telecom-app
```

**Verify port mapping:**
```powershell
docker port telecom-app
```

**Expected output:**
```
8501/tcp -> 0.0.0.0:8501
```

---

### Issue 4: Data Files Not Found

**Error in logs:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'data/raw/master_dataset.csv'
```

**Solution:**
Ensure data files are in the correct location before building:
```
Telco Customer Intelligence Platform/
├── data/
│   └── raw/
│       └── master_dataset.csv  ← Must exist
├── dashboard/
│   └── app.py
└── Dockerfile
```

Then rebuild the image.

---

## 🔄 Updating Your Dashboard

When you make changes to your code:

### Step 1: Stop and Remove Old Container
```powershell
docker stop telecom-app
docker rm telecom-app
```

### Step 2: Rebuild Image
```powershell
docker build -t telecom-dashboard:latest .
```

### Step 3: Run New Container
```powershell
docker run -d -p 8501:8501 --name telecom-app telecom-dashboard:latest
```

---

## 📦 Sharing Your Docker Image

### Option 1: Save to File

**Export image:**
```powershell
docker save -o telecom-dashboard.tar telecom-dashboard:latest
```

**Import on another machine:**
```powershell
docker load -i telecom-dashboard.tar
```

---

### Option 2: Push to Docker Hub

**Step 1: Create Docker Hub account**
- Go to https://hub.docker.com
- Sign up for free account

**Step 2: Login**
```powershell
docker login
```

**Step 3: Tag image with your username**
```powershell
docker tag telecom-dashboard:latest YOUR_USERNAME/telecom-dashboard:latest
```

**Step 4: Push to Docker Hub**
```powershell
docker push YOUR_USERNAME/telecom-dashboard:latest
```

**Step 5: Pull on another machine**
```powershell
docker pull YOUR_USERNAME/telecom-dashboard:latest
docker run -d -p 8501:8501 --name telecom-app YOUR_USERNAME/telecom-dashboard:latest
```

---

## 🌐 Deploying to Cloud

### Deploy to AWS (EC2)

**Step 1: Launch EC2 Instance**
- Choose Ubuntu Server
- Open port 8501 in Security Group

**Step 2: Install Docker on EC2**
```bash
sudo apt update
sudo apt install docker.io -y
sudo systemctl start docker
sudo systemctl enable docker
```

**Step 3: Copy project files or pull from Docker Hub**
```bash
# Option A: Pull from Docker Hub
sudo docker pull YOUR_USERNAME/telecom-dashboard:latest

# Option B: Copy files and build
scp -r "Telco Customer Intelligence Platform" ubuntu@YOUR_EC2_IP:~/
ssh ubuntu@YOUR_EC2_IP
cd "Telco Customer Intelligence Platform"
sudo docker build -t telecom-dashboard:latest .
```

**Step 4: Run container**
```bash
sudo docker run -d -p 8501:8501 --name telecom-app telecom-dashboard:latest
```

**Step 5: Access dashboard**
```
http://YOUR_EC2_PUBLIC_IP:8501
```

---

### Deploy to Azure Container Instances

**Step 1: Install Azure CLI**
```powershell
# Download from: https://aka.ms/installazurecliwindows
```

**Step 2: Login to Azure**
```powershell
az login
```

**Step 3: Create resource group**
```powershell
az group create --name telecom-rg --location eastus
```

**Step 4: Create container registry**
```powershell
az acr create --resource-group telecom-rg --name telecomregistry --sku Basic
```

**Step 5: Push image to registry**
```powershell
az acr login --name telecomregistry
docker tag telecom-dashboard:latest telecomregistry.azurecr.io/telecom-dashboard:latest
docker push telecomregistry.azurecr.io/telecom-dashboard:latest
```

**Step 6: Deploy container**
```powershell
az container create --resource-group telecom-rg --name telecom-app --image telecomregistry.azurecr.io/telecom-dashboard:latest --dns-name-label telecom-dashboard --ports 8501
```

---

## 📊 Monitoring

### View Resource Usage

```powershell
docker stats telecom-app
```

**Output:**
```
CONTAINER ID   NAME          CPU %     MEM USAGE / LIMIT     MEM %     NET I/O
abc123def456   telecom-app   2.50%     250MiB / 8GiB        3.12%     1.2kB / 850B
```

---

## 🔒 Security Best Practices

### 1. Don't Include Sensitive Data in Image
Create `.dockerignore` file:
```
.env
*.log
__pycache__
.pytest_cache
.vscode
```

### 2. Use Environment Variables
For sensitive configuration:
```powershell
docker run -d -p 8501:8501 -e DATABASE_URL=your_url --name telecom-app telecom-dashboard:latest
```

### 3. Run as Non-Root User
Add to Dockerfile:
```dockerfile
RUN useradd -m -u 1000 appuser
USER appuser
```

---

## ✅ Quick Reference

### Build and Run (Fresh Start)
```powershell
docker build -t telecom-dashboard:latest .
docker run -d -p 8501:8501 --name telecom-app telecom-dashboard:latest
```

### Stop and Clean Up
```powershell
docker stop telecom-app
docker rm telecom-app
docker rmi telecom-dashboard:latest
```

### View Dashboard
```
http://localhost:8501
```

---

## 🎯 Success Checklist

- [ ] Docker Desktop installed and running
- [ ] Image built successfully
- [ ] Container running (check with `docker ps`)
- [ ] Dashboard accessible at http://localhost:8501
- [ ] All features working correctly
- [ ] Logs show no errors (`docker logs telecom-app`)

---

## 📞 Need Help?

**Check container logs:**
```powershell
docker logs telecom-app
```

**Access container shell:**
```powershell
docker exec -it telecom-app bash
```

**Restart everything:**
```powershell
docker stop telecom-app
docker rm telecom-app
docker build -t telecom-dashboard:latest .
docker run -d -p 8501:8501 --name telecom-app telecom-dashboard:latest
```

---

**🎉 Congratulations! Your Streamlit dashboard is now running in Docker!**
