# Security Policy

## Reporting a Vulnerability

If you find a security issue in this project, please **do not** open a
public GitHub issue. Instead, report it privately:

- Email: `lf@logosflux.io`
- GitHub: open a [private security advisory](https://github.com/Logos-Flux/spark-voice-pipeline/security/advisories/new).

Please include:

- A description of the issue and its impact
- Steps to reproduce
- The affected versions / commits
- Any proof-of-concept code (if applicable)

We will acknowledge receipt within 7 days and aim to ship a fix or a
mitigation within 30 days for confirmed issues.

## Supported Versions

Only the latest tagged release receives security fixes.

## Known Limitations

The services in this project ship **without built-in authentication**.
By default they bind to `127.0.0.1` (loopback only). Exposing them on
any non-loopback interface (`0.0.0.0`, LAN IP, public IP) without
fronting them with auth (SSH tunnel, Tailscale, authenticated reverse
proxy) lets anyone with network reach drive the GPU, the local LLM, and
the voice pipeline. This is documented in the README; it is not a
vulnerability we will issue an advisory for.

Vulnerabilities in code paths reachable from a properly-fronted
deployment (e.g. command injection through a sanitized header, path
traversal in a file handler) are in scope and welcome.
