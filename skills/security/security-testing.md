---
name: security-testing
description: Implement comprehensive security testing including penetration testing, DAST, fuzzing, and automated security test suites to identify vulnerabilities before production.
tags:
  - security
  - testing
  - penetration-testing
  - dast
  - fuzzing
  - security-automation
category: security
version: 1.0.0
---

# Security Testing

Comprehensive security testing strategies from automated scanning to manual penetration testing, covering DAST, fuzzing, API security testing, and vulnerability validation.

## When to Use This Skill

- **Pre-release Testing**: Validate security before deployment
- **Compliance**: Meet security testing requirements (PCI-DSS, SOC 2)
- **Vulnerability Validation**: Confirm SAST findings are exploitable
- **API Security**: Test authentication, authorization, input validation
- **Penetration Testing**: Simulate real-world attacks
- **Regression Testing**: Ensure fixes don't introduce new issues

## Security Testing Types

### 1. Dynamic Application Security Testing (DAST)

**Purpose**: Black-box testing of running applications

**Tools**:
- **OWASP ZAP**: Open-source web app scanner
- **Burp Suite Pro**: Commercial security testing platform
- **Acunetix**: Automated vulnerability scanner
- **Nikto**: Web server scanner

**OWASP ZAP Automation**:
```bash
# Passive scan (safe, no attacks)
docker run -v $(pwd):/zap/wrk/:rw \
  -t owasp/zap2docker-stable \
  zap-baseline.py \
  -t https://example.com \
  -r zap-report.html

# Active scan (attacks application)
docker run -v $(pwd):/zap/wrk/:rw \
  -t owasp/zap2docker-stable \
  zap-full-scan.py \
  -t https://example.com \
  -r zap-full-report.html

# API scan with OpenAPI spec
docker run -v $(pwd):/zap/wrk/:rw \
  -t owasp/zap2docker-stable \
  zap-api-scan.py \
  -t https://api.example.com/v1/openapi.json \
  -f openapi \
  -r api-scan-report.html
```

**Burp Suite Professional**:
```bash
# Automated scan with Burp
java -jar burpsuite_pro.jar \
  --project-file=security-test.burp \
  --unpause-spider-and-scanner \
  --config-file=burp-config.json

# Export findings
burp-cli --export-issues \
  --format html \
  --output burp-findings.html
```

### 2. API Security Testing

**Authentication Testing**:
```bash
# Test JWT authentication
curl -X POST https://api.example.com/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"password123"}'

# Extract JWT token
TOKEN=$(jq -r '.token' response.json)

# Test token validation
curl https://api.example.com/api/users/me \
  -H "Authorization: Bearer $TOKEN"

# Test expired token
curl https://api.example.com/api/users/me \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."

# Test malformed token
curl https://api.example.com/api/users/me \
  -H "Authorization: Bearer invalid-token-12345"

# Test missing authorization
curl https://api.example.com/api/users/me
```

**Authorization Testing (IDOR)**:
```python
#!/usr/bin/env python3
"""Test for Insecure Direct Object Reference (IDOR) vulnerabilities"""

import requests

# User 1 credentials
user1_token = "eyJhbGciOiJIUzI1NiI..."
user2_token = "eyJhbGciOiJIUzI1NiJ..."

# User 1's resource ID
user1_resource_id = 123

# User 2 attempts to access User 1's resource (IDOR test)
headers = {"Authorization": f"Bearer {user2_token}"}
response = requests.get(
    f"https://api.example.com/api/resources/{user1_resource_id}",
    headers=headers
)

if response.status_code == 200:
    print("[VULN] IDOR vulnerability detected!")
    print(f"User 2 accessed User 1's resource: {user1_resource_id}")
elif response.status_code == 403:
    print("[PASS] Authorization properly enforced")
else:
    print(f"[INFO] Unexpected status code: {response.status_code}")
```

**Rate Limiting Testing**:
```python
#!/usr/bin/env python3
"""Test rate limiting on API endpoints"""

import requests
import time

endpoint = "https://api.example.com/api/users/search"
headers = {"Authorization": "Bearer token123"}

# Send 100 requests rapidly
failures = 0
successes = 0

for i in range(100):
    response = requests.get(endpoint, headers=headers)
    if response.status_code == 429:  # Too Many Requests
        failures += 1
        print(f"Request {i+1}: Rate limited (429)")
        break
    elif response.status_code == 200:
        successes += 1

    time.sleep(0.1)  # 10 requests/second

print(f"\nResults:")
print(f"  Successes: {successes}")
print(f"  Rate limited: {failures}")

if failures == 0:
    print("[VULN] No rate limiting detected!")
else:
    print(f"[PASS] Rate limiting active after {successes} requests")
```

### 3. Fuzzing

**Purpose**: Send malformed/unexpected input to trigger crashes or errors

**Web Fuzzing with ffuf**:
```bash
# Directory fuzzing
ffuf -u https://example.com/FUZZ \
  -w /usr/share/wordlists/dirbuster/directory-list-2.3-medium.txt \
  -fc 404

# Parameter fuzzing
ffuf -u "https://api.example.com/api/users?id=FUZZ" \
  -w /usr/share/wordlists/fuzz.txt \
  -mc 200,500

# Header fuzzing
ffuf -u https://example.com/api/test \
  -H "X-Custom-Header: FUZZ" \
  -w payloads.txt \
  -mc all -fc 400

# POST data fuzzing
ffuf -u https://example.com/api/search \
  -X POST \
  -d '{"query":"FUZZ"}' \
  -H "Content-Type: application/json" \
  -w sqli-payloads.txt \
  -mc 500
```

**API Fuzzing with RESTler**:
```bash
# Compile OpenAPI spec
python restler/restler_bin/restler/Restler.py compile \
  --api_spec openapi.json

# Fuzz API endpoints
python restler/restler_bin/restler/Restler.py fuzz \
  --grammar_file Compile/grammar.py \
  --dictionary_file Compile/dict.json \
  --settings Compile/engine_settings.json \
  --no_ssl
```

**SQL Injection Fuzzing**:
```python
#!/usr/bin/env python3
"""SQL injection fuzzing"""

import requests

sqli_payloads = [
    "' OR '1'='1",
    "' OR '1'='1' --",
    "' OR '1'='1' /*",
    "admin' --",
    "' UNION SELECT NULL, NULL, NULL --",
    "' AND 1=CAST((SELECT version()) AS int) --",
    "'; DROP TABLE users; --",
    "' OR SLEEP(5) --"
]

endpoint = "https://example.com/api/search"

for payload in sqli_payloads:
    data = {"query": payload}
    response = requests.post(endpoint, json=data)

    if response.status_code == 500:
        print(f"[VULN] Payload triggered error: {payload}")
    elif "error" in response.text.lower() or "sql" in response.text.lower():
        print(f"[VULN] SQL error in response: {payload}")
    elif response.elapsed.total_seconds() > 5:
        print(f"[VULN] Time-based SQLi detected: {payload}")
```

### 4. Penetration Testing

**Reconnaissance Phase**:
```bash
# Subdomain enumeration
subfinder -d example.com -o subdomains.txt
amass enum -d example.com -o amass-results.txt

# Port scanning
nmap -sV -sC -oA nmap-scan example.com

# Service fingerprinting
whatweb https://example.com
wappalyzer https://example.com

# SSL/TLS testing
testssl.sh https://example.com

# DNS enumeration
dig example.com ANY
dnsenum example.com
```

**Vulnerability Assessment**:
```bash
# Nuclei template scanning
nuclei -u https://example.com \
  -t cves/ \
  -t vulnerabilities/ \
  -t exposures/ \
  -severity critical,high \
  -o nuclei-findings.txt

# Nessus scan (if available)
nessuscli scan new \
  --targets example.com \
  --name "Security Assessment" \
  --template "basic"

# OpenVAS scan
gvm-cli --gmp-username admin --gmp-password admin \
  tls --hostname scanner.local \
  socket --xml "<create_task><name>Web Scan</name><target id=\"target-id\"/></create_task>"
```

**Exploitation Phase**:
```bash
# SQL injection with sqlmap
sqlmap -u "https://example.com/api/search?q=test" \
  --cookie="session=abc123" \
  --level=5 \
  --risk=3 \
  --batch \
  --dump

# XSS testing with XSStrike
python xsstrike.py -u "https://example.com/search?q=test" \
  --crawl \
  --fuzzer

# Command injection testing
commix -u "https://example.com/api/ping?host=example.com" \
  --level=3 \
  --technique=tbcse
```

### 5. Security Test Automation

**Pytest Security Tests**:
```python
#!/usr/bin/env python3
"""Automated security test suite"""

import pytest
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://api.example.com"

class TestAuthentication:

    def test_login_rate_limiting(self):
        """Test that rate limiting prevents brute force"""
        endpoint = f"{BASE_URL}/auth/login"

        # Attempt 20 rapid logins
        responses = []
        for _ in range(20):
            response = requests.post(endpoint, json={
                "email": "test@example.com",
                "password": "wrong"
            })
            responses.append(response.status_code)

        # Should see 429 (Too Many Requests) after several attempts
        assert 429 in responses, "Rate limiting not enforced"

    def test_invalid_credentials(self):
        """Test that invalid credentials are rejected"""
        response = requests.post(f"{BASE_URL}/auth/login", json={
            "email": "test@example.com",
            "password": "wrong-password"
        })

        assert response.status_code == 401
        assert "token" not in response.json()

    def test_sql_injection_in_login(self):
        """Test SQL injection prevention in login"""
        payloads = ["' OR '1'='1", "admin' --", "' UNION SELECT NULL--"]

        for payload in payloads:
            response = requests.post(f"{BASE_URL}/auth/login", json={
                "email": payload,
                "password": payload
            })

            # Should not return 200 or leak error details
            assert response.status_code != 200
            assert "sql" not in response.text.lower()
            assert "syntax" not in response.text.lower()

class TestAuthorization:

    @pytest.fixture
    def user_tokens(self):
        """Create two different user tokens"""
        # User 1
        response1 = requests.post(f"{BASE_URL}/auth/login", json={
            "email": "user1@example.com",
            "password": "password123"
        })
        token1 = response1.json()["token"]

        # User 2
        response2 = requests.post(f"{BASE_URL}/auth/login", json={
            "email": "user2@example.com",
            "password": "password456"
        })
        token2 = response2.json()["token"]

        return {"user1": token1, "user2": token2}

    def test_idor_vulnerability(self, user_tokens):
        """Test for Insecure Direct Object Reference"""
        # User 1 creates a resource
        response = requests.post(
            f"{BASE_URL}/api/documents",
            headers={"Authorization": f"Bearer {user_tokens['user1']}"},
            json={"title": "Private Document", "content": "Secret"}
        )
        document_id = response.json()["id"]

        # User 2 attempts to access User 1's document
        response = requests.get(
            f"{BASE_URL}/api/documents/{document_id}",
            headers={"Authorization": f"Bearer {user_tokens['user2']}"}
        )

        # Should be denied (403 Forbidden)
        assert response.status_code == 403

    def test_privilege_escalation(self, user_tokens):
        """Test that regular users cannot access admin endpoints"""
        admin_endpoints = [
            "/api/admin/users",
            "/api/admin/settings",
            "/api/admin/audit-logs"
        ]

        for endpoint in admin_endpoints:
            response = requests.get(
                f"{BASE_URL}{endpoint}",
                headers={"Authorization": f"Bearer {user_tokens['user1']}"}
            )

            # Should be forbidden
            assert response.status_code in [403, 404]

class TestInputValidation:

    def test_xss_prevention(self):
        """Test XSS payload sanitization"""
        xss_payloads = [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "javascript:alert('XSS')"
        ]

        for payload in xss_payloads:
            response = requests.post(f"{BASE_URL}/api/comments", json={
                "content": payload
            })

            # Verify payload is sanitized
            saved_comment = response.json()["content"]
            assert "<script>" not in saved_comment
            assert "javascript:" not in saved_comment

    def test_path_traversal(self):
        """Test path traversal prevention"""
        payloads = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd"
        ]

        for payload in payloads:
            response = requests.get(f"{BASE_URL}/api/files/{payload}")

            # Should not expose file system
            assert response.status_code in [400, 403, 404]
            assert "root:" not in response.text

class TestSecurityHeaders:

    def test_security_headers_present(self):
        """Test that security headers are set"""
        response = requests.get(BASE_URL)
        headers = response.headers

        required_headers = {
            "Strict-Transport-Security": "max-age=31536000",
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": ["DENY", "SAMEORIGIN"],
            "Content-Security-Policy": "default-src",
            "Referrer-Policy": "strict-origin-when-cross-origin"
        }

        for header, expected in required_headers.items():
            assert header in headers, f"Missing header: {header}"

            if isinstance(expected, list):
                assert any(val in headers[header] for val in expected)
            elif isinstance(expected, str):
                assert expected in headers[header]

    def test_sensitive_headers_removed(self):
        """Test that sensitive headers are not exposed"""
        response = requests.get(BASE_URL)
        headers = response.headers

        forbidden_headers = [
            "X-Powered-By",
            "Server",
            "X-AspNet-Version"
        ]

        for header in forbidden_headers:
            assert header not in headers, f"Sensitive header exposed: {header}"

class TestCryptography:

    def test_https_enforced(self):
        """Test that HTTPS is enforced"""
        response = requests.get("http://example.com", allow_redirects=False)

        # Should redirect to HTTPS
        assert response.status_code in [301, 302, 307, 308]
        assert response.headers["Location"].startswith("https://")

    def test_weak_tls_disabled(self):
        """Test that TLS 1.0 and 1.1 are disabled"""
        import ssl
        import socket

        hostname = "example.com"
        port = 443

        weak_protocols = [ssl.PROTOCOL_TLSv1, ssl.PROTOCOL_TLSv1_1]

        for protocol in weak_protocols:
            try:
                context = ssl.SSLContext(protocol)
                sock = socket.create_connection((hostname, port))
                ssl_sock = context.wrap_socket(sock, server_hostname=hostname)
                ssl_sock.close()
                pytest.fail(f"Weak protocol {protocol} is enabled")
            except (ssl.SSLError, OSError):
                pass  # Expected - protocol should be disabled
```

**GitHub Actions Integration**:
```yaml
name: Security Testing

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 2 * * 1'  # Weekly on Monday 2 AM

jobs:
  sast:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Run Semgrep
        uses: returntocorp/semgrep-action@v1
        with:
          config: >-
            p/security-audit
            p/owasp-top-ten

  dast:
    runs-on: ubuntu-latest
    needs: deploy-staging
    steps:
      - name: OWASP ZAP Scan
        uses: zaproxy/action-full-scan@v0.4.0
        with:
          target: 'https://staging.example.com'
          rules_file_name: '.zap/rules.tsv'
          cmd_options: '-a'

      - name: Upload ZAP Report
        uses: actions/upload-artifact@v3
        with:
          name: zap-report
          path: report_html.html

  api-security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Install Dependencies
        run: |
          pip install pytest requests

      - name: Run API Security Tests
        run: |
          pytest tests/security/ -v --html=security-report.html

      - name: Upload Test Report
        uses: actions/upload-artifact@v3
        with:
          name: security-test-report
          path: security-report.html

  dependency-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Run Snyk
        uses: snyk/actions/node@master
        env:
          SNYK_TOKEN: ${{ secrets.SNYK_TOKEN }}
        with:
          args: --severity-threshold=high
```

## Security Testing Checklist

```markdown
## Authentication & Authorization
- [ ] Test rate limiting on login endpoint
- [ ] Test account lockout after failed attempts
- [ ] Test password complexity requirements
- [ ] Test session timeout
- [ ] Test logout functionality
- [ ] Test IDOR vulnerabilities
- [ ] Test privilege escalation
- [ ] Test JWT token validation
- [ ] Test OAuth flow security

## Input Validation
- [ ] Test SQL injection in all parameters
- [ ] Test XSS in all input fields
- [ ] Test command injection
- [ ] Test path traversal
- [ ] Test XXE (XML External Entity)
- [ ] Test SSRF (Server-Side Request Forgery)
- [ ] Test file upload restrictions
- [ ] Test mass assignment

## Cryptography
- [ ] Test HTTPS enforcement
- [ ] Test TLS version (1.2+ only)
- [ ] Test certificate validity
- [ ] Test weak ciphers disabled
- [ ] Test password hashing (bcrypt, Argon2)
- [ ] Test sensitive data encryption

## Security Headers
- [ ] Test Content-Security-Policy
- [ ] Test X-Content-Type-Options
- [ ] Test X-Frame-Options
- [ ] Test Strict-Transport-Security
- [ ] Test Referrer-Policy
- [ ] Test Permissions-Policy

## Business Logic
- [ ] Test race conditions
- [ ] Test workflow bypass
- [ ] Test negative values
- [ ] Test boundary conditions
- [ ] Test state tampering

## API Security
- [ ] Test API rate limiting
- [ ] Test API key validation
- [ ] Test CORS configuration
- [ ] Test API versioning
- [ ] Test error messages (no info leakage)
```

## Tools Reference

### Open-Source Tools
```yaml
DAST:
  - OWASP ZAP: https://www.zaproxy.org/
  - Nikto: https://cirt.net/Nikto2
  - w3af: http://w3af.org/

Fuzzing:
  - ffuf: https://github.com/ffuf/ffuf
  - wfuzz: https://github.com/xmendez/wfuzz
  - RESTler: https://github.com/microsoft/restler-fuzzer

SQL Injection:
  - sqlmap: https://sqlmap.org/
  - NoSQLMap: https://github.com/codingo/NoSQLMap

Subdomain Enumeration:
  - Subfinder: https://github.com/projectdiscovery/subfinder
  - Amass: https://github.com/OWASP/Amass

Port Scanning:
  - nmap: https://nmap.org/
  - masscan: https://github.com/robertdavidgraham/masscan

Vulnerability Scanning:
  - Nuclei: https://github.com/projectdiscovery/nuclei
  - OpenVAS: https://www.openvas.org/
```

### Commercial Tools
```yaml
DAST:
  - Burp Suite Professional: https://portswigger.net/burp/pro
  - Acunetix: https://www.acunetix.com/
  - Veracode DAST: https://www.veracode.com/

Penetration Testing:
  - Metasploit Pro: https://www.metasploit.com/
  - Core Impact: https://www.coresecurity.com/
  - Cobalt Strike: https://www.cobaltstrike.com/

Vulnerability Management:
  - Nessus Professional: https://www.tenable.com/products/nessus
  - Qualys VMDR: https://www.qualys.com/
  - Rapid7 InsightVM: https://www.rapid7.com/
```

## Best Practices

1. **Test Early, Test Often**: Integrate security testing in CI/CD
2. **Combine Approaches**: Use SAST, DAST, and manual testing
3. **Prioritize by Risk**: Focus on critical/high severity first
4. **Validate Findings**: Confirm vulnerabilities are exploitable
5. **Retest After Fixes**: Ensure patches are effective
6. **Document Everything**: Maintain test cases and results
7. **Automate Where Possible**: Save time on regression testing
8. **Stay Current**: Update tools and vulnerability databases
9. **Train Developers**: Share findings and prevention techniques
10. **Continuous Improvement**: Learn from each security test

Security testing is not a one-time activity - it's a continuous process integrated throughout the development lifecycle. Use these techniques to proactively identify and remediate vulnerabilities before attackers do.
