---
name: network-security
description: Implement network security controls including firewalls, network segmentation, TLS/SSL, VPNs, and zero-trust architecture to protect applications and infrastructure.
trigger_keywords: [network security, firewall, iptables, tls, ssl, vpn, wireguard, zero trust, network segmentation, vlan, ids, ips, suricata]
tags:
  - security
  - network-security
  - firewalls
  - zero-trust
  - tls-ssl
  - network-segmentation
category: security
version: 1.0.0
---

# Network Security

Comprehensive network security implementation covering firewalls, network segmentation, TLS/SSL configuration, VPNs, intrusion detection, and zero-trust architecture.

## When to Use This Skill

- **Infrastructure Setup**: Secure network architecture design
- **Cloud Migration**: Implement network security in cloud environments
- **Compliance**: Meet network security requirements (PCI-DSS, HIPAA)
- **Zero Trust**: Implement zero-trust network architecture
- **Incident Response**: Contain network-based attacks
- **Security Hardening**: Strengthen network defenses

## Core Concepts

- **Defense in Depth**: Network security requires multiple overlapping controls - perimeter firewalls, network segmentation, host-based firewalls, IDS/IPS, and TLS encryption. Assume any single layer can be bypassed.

- **Zero Trust Model**: Never trust, always verify. All network traffic should be authenticated and encrypted, regardless of source. Microsegmentation limits lateral movement even if perimeter is breached.

- **Encryption in Transit**: All data in motion must use TLS 1.2+ (preferably 1.3). Disable legacy protocols (SSLv3, TLS 1.0/1.1). Use HSTS headers and certificate pinning for critical applications.

- **Network Segmentation**: Separate networks by trust level (DMZ, internal, database tier). Use VLANs, security groups, and firewall rules to enforce boundaries. Default deny all, explicitly allow required traffic.

- **Monitoring and Detection**: Deploy IDS/IPS (Suricata, Snort) to detect malicious traffic. Log all firewall decisions. Alert on anomalous traffic patterns. Maintain network flow records for forensics.

## Network Security Layers

### 1. Network Segmentation

**Purpose**: Divide network into isolated segments to limit attack surface and contain breaches

**VLAN Configuration**:
```yaml
# Network topology with VLANs
Network Architecture:
  DMZ (VLAN 10):
    - Web servers (public-facing)
    - Load balancers
    - API gateways
    - Subnet: 10.0.10.0/24

  Application Tier (VLAN 20):
    - Application servers
    - Microservices
    - Internal APIs
    - Subnet: 10.0.20.0/24

  Data Tier (VLAN 30):
    - Database servers
    - File storage
    - Backup systems
    - Subnet: 10.0.30.0/24

  Management (VLAN 40):
    - Bastion hosts
    - Monitoring systems
    - Admin workstations
    - Subnet: 10.0.40.0/24

Traffic Flow Rules:
  DMZ → Application: Allowed on specific ports (8080, 8443)
  Application → Data: Allowed on database ports (5432, 3306)
  Data → DMZ: Denied (databases never initiate outbound to DMZ)
  Management → All: Allowed (admin access)
  All → Management: Denied (no one can initiate to management)
```

**AWS VPC Security Groups**:
```yaml
# Web Tier Security Group
WebTierSG:
  Inbound:
    - Source: 0.0.0.0/0
      Protocol: TCP
      Port: 443
      Description: HTTPS from internet

    - Source: 0.0.0.0/0
      Protocol: TCP
      Port: 80
      Description: HTTP (redirect to HTTPS)

  Outbound:
    - Destination: AppTierSG
      Protocol: TCP
      Port: 8080
      Description: To application servers

# Application Tier Security Group
AppTierSG:
  Inbound:
    - Source: WebTierSG
      Protocol: TCP
      Port: 8080
      Description: From web tier

    - Source: BastionSG
      Protocol: TCP
      Port: 22
      Description: SSH from bastion

  Outbound:
    - Destination: DataTierSG
      Protocol: TCP
      Port: 5432
      Description: To PostgreSQL database

    - Destination: 0.0.0.0/0
      Protocol: TCP
      Port: 443
      Description: HTTPS for external APIs

# Data Tier Security Group
DataTierSG:
  Inbound:
    - Source: AppTierSG
      Protocol: TCP
      Port: 5432
      Description: PostgreSQL from app tier

    - Source: BastionSG
      Protocol: TCP
      Port: 5432
      Description: Database admin access

  Outbound:
    - Destination: 0.0.0.0/0
      Protocol: TCP
      Port: 443
      Description: For backup to S3
```

**Terraform Implementation**:
```hcl
# VPC with multiple subnets
resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "production-vpc"
  }
}

# Public subnet (DMZ)
resource "aws_subnet" "public" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.0.10.0/24"
  availability_zone       = "us-east-1a"
  map_public_ip_on_launch = true

  tags = {
    Name = "public-subnet-dmz"
    Tier = "DMZ"
  }
}

# Private subnet (Application tier)
resource "aws_subnet" "private_app" {
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.0.20.0/24"
  availability_zone = "us-east-1a"

  tags = {
    Name = "private-subnet-app"
    Tier = "Application"
  }
}

# Private subnet (Data tier)
resource "aws_subnet" "private_data" {
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.0.30.0/24"
  availability_zone = "us-east-1a"

  tags = {
    Name = "private-subnet-data"
    Tier = "Data"
  }
}

# Security group for web tier
resource "aws_security_group" "web" {
  name        = "web-tier-sg"
  description = "Security group for web tier"
  vpc_id      = aws_vpc.main.id

  # HTTPS inbound
  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "HTTPS from internet"
  }

  # HTTP inbound (redirect to HTTPS)
  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "HTTP redirect"
  }

  # Outbound to app tier
  egress {
    from_port       = 8080
    to_port         = 8080
    protocol        = "tcp"
    security_groups = [aws_security_group.app.id]
    description     = "To application tier"
  }

  tags = {
    Name = "web-tier-sg"
  }
}
```

### 2. Firewall Configuration

**iptables (Linux Firewall)**:
```bash
#!/bin/bash
# Secure iptables configuration

# Flush existing rules
iptables -F
iptables -X

# Default policies: DROP everything
iptables -P INPUT DROP
iptables -P FORWARD DROP
iptables -P OUTPUT DROP

# Allow loopback
iptables -A INPUT -i lo -j ACCEPT
iptables -A OUTPUT -o lo -j ACCEPT

# Allow established connections
iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT
iptables -A OUTPUT -m state --state ESTABLISHED,RELATED -j ACCEPT

# Allow SSH (rate limited)
iptables -A INPUT -p tcp --dport 22 -m state --state NEW -m recent --set --name SSH
iptables -A INPUT -p tcp --dport 22 -m state --state NEW -m recent --update --seconds 60 --hitcount 4 --rttl --name SSH -j DROP
iptables -A INPUT -p tcp --dport 22 -j ACCEPT

# Allow HTTP/HTTPS
iptables -A INPUT -p tcp --dport 80 -j ACCEPT
iptables -A INPUT -p tcp --dport 443 -j ACCEPT

# Allow DNS queries
iptables -A OUTPUT -p udp --dport 53 -j ACCEPT
iptables -A OUTPUT -p tcp --dport 53 -j ACCEPT

# Allow NTP
iptables -A OUTPUT -p udp --dport 123 -j ACCEPT

# Allow HTTPS outbound (for API calls)
iptables -A OUTPUT -p tcp --dport 443 -j ACCEPT

# Log dropped packets (debugging)
iptables -A INPUT -j LOG --log-prefix "IPT-INPUT-DROP: " --log-level 4
iptables -A OUTPUT -j LOG --log-prefix "IPT-OUTPUT-DROP: " --log-level 4

# Drop invalid packets
iptables -A INPUT -m state --state INVALID -j DROP

# Protection against port scanning
iptables -A INPUT -p tcp --tcp-flags ALL NONE -j DROP
iptables -A INPUT -p tcp --tcp-flags ALL ALL -j DROP

# Save rules
iptables-save > /etc/iptables/rules.v4
```

**AWS Network ACL**:
```hcl
resource "aws_network_acl" "app_tier" {
  vpc_id     = aws_vpc.main.id
  subnet_ids = [aws_subnet.private_app.id]

  # Inbound: Allow from web tier
  ingress {
    rule_no    = 100
    protocol   = "tcp"
    action     = "allow"
    cidr_block = "10.0.10.0/24"
    from_port  = 8080
    to_port    = 8080
  }

  # Inbound: Allow return traffic
  ingress {
    rule_no    = 110
    protocol   = "tcp"
    action     = "allow"
    cidr_block = "0.0.0.0/0"
    from_port  = 1024
    to_port    = 65535
  }

  # Outbound: Allow to data tier
  egress {
    rule_no    = 100
    protocol   = "tcp"
    action     = "allow"
    cidr_block = "10.0.30.0/24"
    from_port  = 5432
    to_port    = 5432
  }

  # Outbound: Allow return traffic
  egress {
    rule_no    = 110
    protocol   = "tcp"
    action     = "allow"
    cidr_block = "0.0.0.0/0"
    from_port  = 1024
    to_port    = 65535
  }

  tags = {
    Name = "app-tier-nacl"
  }
}
```

### 3. TLS/SSL Configuration

**Nginx TLS Best Practices**:
```nginx
# /etc/nginx/sites-available/secure-app
server {
    listen 80;
    server_name example.com www.example.com;

    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name example.com www.example.com;

    # SSL certificates
    ssl_certificate /etc/letsencrypt/live/example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/example.com/privkey.pem;

    # SSL protocols (TLS 1.2 and 1.3 only)
    ssl_protocols TLSv1.2 TLSv1.3;

    # Strong cipher suites (forward secrecy)
    ssl_ciphers 'ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-ECDSA-CHACHA20-POLY1305:ECDHE-RSA-CHACHA20-POLY1305';
    ssl_prefer_server_ciphers off;

    # SSL session settings
    ssl_session_timeout 1d;
    ssl_session_cache shared:SSL:50m;
    ssl_session_tickets off;

    # OCSP stapling
    ssl_stapling on;
    ssl_stapling_verify on;
    resolver 8.8.8.8 8.8.4.4 valid=300s;
    resolver_timeout 5s;

    # Security headers
    add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload" always;
    add_header X-Frame-Options "DENY" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline' https://cdn.example.com; style-src 'self' 'unsafe-inline';" always;

    # Diffie-Hellman parameter
    ssl_dhparam /etc/nginx/ssl/dhparam.pem;

    location / {
        proxy_pass http://localhost:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

**Generate Strong DH Parameters**:
```bash
# Generate 4096-bit DH parameters (takes time)
openssl dhparam -out /etc/nginx/ssl/dhparam.pem 4096
```

**Let's Encrypt Auto-Renewal**:
```bash
# Install certbot
apt-get install certbot python3-certbot-nginx

# Obtain certificate
certbot --nginx -d example.com -d www.example.com

# Auto-renewal (runs daily)
systemctl enable certbot.timer
systemctl start certbot.timer

# Test renewal
certbot renew --dry-run
```

**TLS Testing**:
```bash
# Test TLS configuration
testssl.sh https://example.com

# Check certificate
echo | openssl s_client -connect example.com:443 -servername example.com 2>/dev/null | openssl x509 -noout -text

# Verify OCSP stapling
echo | openssl s_client -connect example.com:443 -servername example.com -status 2>/dev/null | grep -A 17 'OCSP'
```

### 4. VPN Configuration

**WireGuard VPN Setup**:
```bash
# Install WireGuard
apt-get install wireguard

# Generate server keys
wg genkey | tee /etc/wireguard/server_private.key | wg pubkey > /etc/wireguard/server_public.key
chmod 600 /etc/wireguard/server_private.key

# Server configuration
cat > /etc/wireguard/wg0.conf <<EOF
[Interface]
PrivateKey = $(cat /etc/wireguard/server_private.key)
Address = 10.8.0.1/24
ListenPort = 51820
PostUp = iptables -A FORWARD -i wg0 -j ACCEPT; iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE
PostDown = iptables -D FORWARD -i wg0 -j ACCEPT; iptables -t nat -D POSTROUTING -o eth0 -j MASQUERADE

# Client 1
[Peer]
PublicKey = CLIENT1_PUBLIC_KEY
AllowedIPs = 10.8.0.2/32

# Client 2
[Peer]
PublicKey = CLIENT2_PUBLIC_KEY
AllowedIPs = 10.8.0.3/32
EOF

# Enable IP forwarding
echo "net.ipv4.ip_forward=1" >> /etc/sysctl.conf
sysctl -p

# Start WireGuard
systemctl enable wg-quick@wg0
systemctl start wg-quick@wg0
```

**Client Configuration**:
```ini
[Interface]
PrivateKey = CLIENT_PRIVATE_KEY
Address = 10.8.0.2/24
DNS = 10.8.0.1

[Peer]
PublicKey = SERVER_PUBLIC_KEY
Endpoint = vpn.example.com:51820
AllowedIPs = 10.0.0.0/8, 172.16.0.0/12
PersistentKeepalive = 25
```

### 5. Intrusion Detection/Prevention (IDS/IPS)

**Suricata IDS Setup**:
```yaml
# /etc/suricata/suricata.yaml
vars:
  address-groups:
    HOME_NET: "[10.0.0.0/8,172.16.0.0/12,192.168.0.0/16]"
    EXTERNAL_NET: "!$HOME_NET"

default-rule-path: /etc/suricata/rules

rule-files:
  - suricata.rules
  - emerging-threats.rules

outputs:
  - fast:
      enabled: yes
      filename: fast.log
      append: yes

  - eve-log:
      enabled: yes
      filetype: regular
      filename: eve.json
      types:
        - alert:
            payload: yes
            metadata: yes
        - http:
            extended: yes
        - dns:
            query: yes
            answer: yes
        - tls:
            extended: yes

af-packet:
  - interface: eth0
    threads: auto
    cluster-id: 99
    cluster-type: cluster_flow
    defrag: yes
```

**Custom Suricata Rules**:
```
# /etc/suricata/rules/custom.rules

# Detect SQL injection attempts
alert http any any -> $HOME_NET any (msg:"SQL Injection Attempt"; flow:established,to_server; content:"SELECT"; nocase; content:"FROM"; nocase; content:"'"; sid:1000001; rev:1;)

# Detect XSS attempts
alert http any any -> $HOME_NET any (msg:"XSS Attempt"; flow:established,to_server; content:"<script>"; nocase; sid:1000002; rev:1;)

# Detect command injection
alert http any any -> $HOME_NET any (msg:"Command Injection Attempt"; flow:established,to_server; pcre:"/(\||;|`|$\(|&&)/"; sid:1000003; rev:1;)

# Detect brute force SSH
alert ssh any any -> $HOME_NET 22 (msg:"SSH Brute Force Attempt"; flow:to_server; threshold:type both, track by_src, count 5, seconds 60; sid:1000004; rev:1;)
```

**Fail2Ban (Automated Response)**:
```ini
# /etc/fail2ban/jail.local
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 5
backend = systemd

[sshd]
enabled = true
port = ssh
logpath = /var/log/auth.log

[nginx-http-auth]
enabled = true
port = http,https
logpath = /var/log/nginx/error.log

[nginx-limit-req]
enabled = true
port = http,https
logpath = /var/log/nginx/error.log

[nginx-noscript]
enabled = true
port = http,https
logpath = /var/log/nginx/access.log
```

### 6. Zero Trust Architecture

**Principles**:
```yaml
Zero Trust Principles:
  1. Verify Explicitly:
     - Always authenticate and authorize
     - Use all available data points (identity, location, device, workload, data)

  2. Use Least Privilege Access:
     - Just-in-time (JIT) access
     - Just-enough-access (JEA)
     - Risk-based adaptive policies

  3. Assume Breach:
     - Minimize blast radius
     - Segment access
     - Verify end-to-end encryption
     - Use analytics for visibility and threat detection
```

**Implementation with Service Mesh (Istio)**:
```yaml
# Mutual TLS between services
apiVersion: security.istio.io/v1beta1
kind: PeerAuthentication
metadata:
  name: default
  namespace: production
spec:
  mtls:
    mode: STRICT

---
# Authorization policy
apiVersion: security.istio.io/v1beta1
kind: AuthorizationPolicy
metadata:
  name: frontend-to-backend
  namespace: production
spec:
  selector:
    matchLabels:
      app: backend
  action: ALLOW
  rules:
  - from:
    - source:
        principals: ["cluster.local/ns/production/sa/frontend"]
    to:
    - operation:
        methods: ["GET", "POST"]
        paths: ["/api/*"]
    when:
    - key: request.auth.claims[iss]
      values: ["https://auth.example.com"]

---
# Request authentication (JWT validation)
apiVersion: security.istio.io/v1beta1
kind: RequestAuthentication
metadata:
  name: jwt-auth
  namespace: production
spec:
  selector:
    matchLabels:
      app: backend
  jwtRules:
  - issuer: "https://auth.example.com"
    jwksUri: "https://auth.example.com/.well-known/jwks.json"
    audiences:
    - "api.example.com"
```

## Network Security Checklist

```markdown
## Network Architecture
- [ ] Network segmentation implemented (DMZ, App, Data tiers)
- [ ] VLANs configured for isolation
- [ ] Network ACLs in place
- [ ] Security groups with least privilege
- [ ] Bastion hosts for administrative access
- [ ] No direct internet access to databases

## Firewalls
- [ ] Host-based firewalls enabled (iptables, Windows Firewall)
- [ ] Network firewalls configured (AWS Security Groups, Azure NSG)
- [ ] Default deny policies
- [ ] Rate limiting on critical services
- [ ] Log all dropped packets
- [ ] Regular firewall rule audits

## TLS/SSL
- [ ] TLS 1.2+ enforced (TLS 1.0/1.1 disabled)
- [ ] Strong cipher suites configured
- [ ] HTTPS enforced (HTTP redirects)
- [ ] Valid SSL certificates (not expired)
- [ ] HSTS header enabled
- [ ] Certificate pinning (mobile apps)
- [ ] OCSP stapling enabled

## VPN & Remote Access
- [ ] VPN required for remote access
- [ ] Strong VPN encryption (WireGuard, OpenVPN)
- [ ] Multi-factor authentication for VPN
- [ ] IP allowlisting where possible
- [ ] VPN access logged and monitored

## Intrusion Detection
- [ ] IDS/IPS deployed (Suricata, Snort)
- [ ] Log aggregation and monitoring (SIEM)
- [ ] Automated response (Fail2Ban)
- [ ] Regular rule updates
- [ ] Alert thresholds configured

## Monitoring & Logging
- [ ] Network traffic logged
- [ ] Connection attempts logged
- [ ] Failed authentication logged
- [ ] Centralized log storage
- [ ] Real-time alerting configured
- [ ] Log retention policy implemented
```

## Network Security Tools

```yaml
Scanning & Discovery:
  - nmap: Network scanning and port discovery
  - masscan: Fast port scanner
  - Shodan: Internet-connected device search

Packet Analysis:
  - Wireshark: Packet capture and analysis
  - tcpdump: Command-line packet analyzer
  - Zeek: Network security monitoring

Intrusion Detection:
  - Suricata: IDS/IPS engine
  - Snort: IDS/IPS
  - OSSEC: Host-based IDS

VPN:
  - WireGuard: Modern VPN protocol
  - OpenVPN: Mature VPN solution
  - IPsec: Industry standard VPN

Firewalls:
  - iptables: Linux firewall
  - pfSense: Open-source firewall
  - OPNsense: Firewall and router

Monitoring:
  - Prometheus: Metrics collection
  - Grafana: Visualization
  - ELK Stack: Log aggregation
```

## Best Practices

1. **Defense in Depth**: Multiple layers of security
2. **Least Privilege**: Minimal necessary access
3. **Zero Trust**: Never trust, always verify
4. **Segmentation**: Isolate sensitive systems
5. **Encryption**: TLS for all communications
6. **Monitoring**: Log and alert on anomalies
7. **Patch Management**: Keep systems updated
8. **Incident Response**: Have a plan ready
9. **Regular Audits**: Periodic security reviews
10. **Documentation**: Maintain network diagrams

Network security is fundamental to protecting applications and data. Implement these controls to build a robust, defense-in-depth network architecture that can withstand modern threats.
