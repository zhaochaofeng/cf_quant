# 服务器监控部署手册

**部署时间**: 2026-02-06  
**监控范围**: CPU、内存、磁盘、网络、GPU (RTX 3090)  
**部署架构**: Prometheus + Node Exporter + DCGM Exporter + Grafana

---

## 一、环境要求

| 项目 | 版本/要求 |
|------|----------|
| 操作系统 | Linux (CentOS/RHEL/Ubuntu) |
| Docker | 已安装并配置 NVIDIA Container Toolkit |
| NVIDIA 驱动 | 已安装 (验证: nvidia-smi) |
| 端口占用 | 9090, 9100, 9400 可用 |

---

## 二、部署步骤

### 1. 创建部署目录

```bash
mkdir -p ~/monitoring
cd ~/monitoring
```

### 2. 创建 docker-compose.yml

```yaml
services:
  prometheus:
    image: prom/prometheus:v2.47.0
    container_name: prometheus
    restart: unless-stopped
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=30d'
      - '--web.enable-lifecycle'
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
      - /etc/localtime:/etc/localtime:ro
    ports:
      - "9090:9090"
    networks:
      - monitoring
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G

  node-exporter:
    image: prom/node-exporter:v1.6.1
    container_name: node-exporter
    restart: unless-stopped
    privileged: true
    user: root
    command:
      - '--path.procfs=/host/proc'
      - '--path.rootfs=/rootfs'
      - '--path.sysfs=/host/sys'
      - '--collector.filesystem.mount-points-exclude=^/(sys|proc|dev|host|etc)($$|/)'
    volumes:
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
      - /:/rootfs:ro
      - /etc/localtime:/etc/localtime:ro
    ports:
      - "9100:9100"
    networks:
      - monitoring

  dcgm-exporter:
    image: nvidia/dcgm-exporter:3.1.8-3.1.5-ubuntu20.04
    container_name: dcgm-exporter
    restart: unless-stopped
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    ports:
      - "9400:9400"
    networks:
      - monitoring

volumes:
  prometheus_data:
    driver: local

networks:
  monitoring:
    driver: bridge
```

### 3. 创建 prometheus.yml

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']
    scrape_interval: 10s

  - job_name: 'nvidia-dcgm'
    static_configs:
      - targets: ['dcgm-exporter:9400']
    scrape_interval: 5s
```

### 4. 配置 Docker NVIDIA Runtime

**文件**: `/etc/docker/daemon.json`

添加以下内容：
```json
{
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  }
}
```

**重启 Docker**:
```bash
sudo systemctl restart docker
```

### 5. 启动服务

```bash
cd ~/monitoring
docker compose up -d
```

---

## 三、验证部署

### 1. 检查容器状态

```bash
docker ps | grep -E 'prometheus|exporter'
```

**预期输出**:
```
prom/prometheus:v2.47.0    Up X minutes    0.0.0.0:9090->9090/tcp    prometheus
prom/node-exporter:v1.6.1  Up X minutes    0.0.0.0:9100->9100/tcp    node-exporter
nvidia/dcgm-exporter       Up X minutes    0.0.0.0:9400->9400/tcp    dcgm-exporter
```

### 2. 验证 Prometheus 目标

访问: `http://服务器IP:9090/api/v1/targets`

**预期状态**:
- node-exporter: UP
- nvidia-dcgm: UP  
- prometheus: UP

### 3. 验证 GPU 指标

```bash
curl http://localhost:9400/metrics | grep DCGM_FI_DEV_GPU_UTIL
```

---

## 四、Grafana 配置

### 1. 添加 Prometheus 数据源

| 配置项 | 值 |
|--------|-----|
| URL | `http://localhost:9090` |
| Authentication | No Authentication |
| 其他选项 | 默认 |

点击 **Save & Test** → 显示 "Data source is working"

### 2. 导入 Dashboard

#### 主机监控 (ID: 1860)
1. 点击 **+** → **Import**
2. 输入: `1860`
3. 选择 Prometheus 数据源
4. 点击 **Import**

**覆盖指标**: CPU、内存、磁盘、网络、系统负载

#### GPU 监控 (ID: 12239)
1. 点击 **+** → **Import**
2. 输入: `12239`
3. 选择 Prometheus 数据源
4. 点击 **Import**

**覆盖指标**: GPU利用率、显存、温度、功耗、频率

---

## 五、访问地址

| 服务 | URL | 用途 |
|------|-----|------|
| Prometheus | `http://服务器IP:9090` | 数据查询/调试 |
| Node Exporter | `http://服务器IP:9100/metrics` | 主机原始指标 |
| DCGM Exporter | `http://服务器IP:9400/metrics` | GPU原始指标 |
| Grafana | 你的现有Grafana地址 | 可视化看板 |

---

## 六、常用 PromQL 查询

### CPU 使用率
```promql
100 - (avg(irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)
```

### 内存使用率
```promql
(node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes * 100
```

### GPU 利用率
```promql
DCGM_FI_DEV_GPU_UTIL
```

### 显存使用率
```promql
DCGM_FI_DEV_FB_USED / (DCGM_FI_DEV_FB_USED + DCGM_FI_DEV_FB_FREE) * 100
```

---

## 七、故障排查

### 问题1: Prometheus 数据目录权限错误
**现象**: `permission denied`  
**解决**: 确保容器以 root 运行，或正确设置 volume 权限

### 问题2: DCGM 启动失败 "unknown runtime"
**现象**: `unknown or invalid runtime name: nvidia`  
**解决**: 配置 `/etc/docker/daemon.json` 添加 nvidia runtime，重启 Docker

### 问题3: Grafana 无法连接 Prometheus
**现象**: `Data source error`  
**解决**: 
- 检查 Prometheus 容器状态: `docker ps`
- 检查网络连通性: `curl http://localhost:9090/api/v1/targets`
- 确认 URL 正确（使用 IP 而非 localhost 如果 Grafana 在容器中）

---

## 八、后续维护

### 查看日志
```bash
docker logs prometheus
docker logs node-exporter
docker logs dcgm-exporter
```

### 重启服务
```bash
cd ~/monitoring
docker compose restart
```

### 停止服务
```bash
cd ~/monitoring
docker compose down
```

### 更新镜像
```bash
cd ~/monitoring
docker compose pull
docker compose up -d
```

---

**部署完成时间**: 2026-02-06  
**文档版本**: v1.0
