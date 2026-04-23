# Security Policy

## Supported versions

| Version | Supported |
|---------|-----------|
| latest  | Yes       |
| < latest | No       |

Only the latest release receives security updates. Users should always run the most recent version.

## Reporting a vulnerability

**Do not open a public issue for security vulnerabilities.**

Please report vulnerabilities by emailing the maintainer directly. Include:

1. A description of the vulnerability.
2. Steps to reproduce the issue.
3. The potential impact.
4. Any suggested fix or mitigation.

You should receive an acknowledgment within 48 hours. We will work with you to understand the issue and coordinate a fix before any public disclosure.

## Disclosure policy

- We will acknowledge receipt within 48 hours.
- We will provide an initial assessment within 7 days.
- We will release a fix as soon as practical, typically within 30 days.
- We will credit the reporter in the release notes (unless anonymity is requested).
- **No public disclosure before a fix is available**, unless 90 days have elapsed without resolution.

## Security considerations

- **GGUF parsing:** All GGUF file parsing runs through `GGUFValidator`, which checks declared tensor sizes against physical RAM before allocation to prevent memory exhaustion attacks.
- **API authentication:** Bearer token auth is available via `BearerAuthMiddleware`. Always enable it when exposing the server beyond localhost.
- **Dependencies:** `pip-audit --strict` runs in CI on every PR to catch known CVEs in resolved dependencies.
- **Wired limit script:** The `set_wired_limit.sh` script validates input to prevent command injection. On CI runners, it is installed as a separate copy (not from the checked-out repo) and restricted via sudoers.
