# Security policy

## Reporting a vulnerability

Please do not disclose credentials, exploit details, or sensitive data in a public issue. If GitHub private vulnerability reporting is enabled for this repository, use the repository's **Security → Advisories** workflow. Otherwise, contact the repository owner privately through GitHub before sharing technical details.

Include the affected file or component, the impact, and a minimal reproduction when it is safe to do so. Do not include live credentials; revoke and rotate any credential that may already have been exposed.

## Supported versions

Security fixes target the current `main` branch. This project is an experimental research toolkit and is not a production model-serving service.

## Automated code scanning

The repository includes a GitHub Actions CodeQL workflow for the supported Python code and workflow definitions. Results are uploaded to GitHub Code Scanning when that feature is available and enabled for the repository. Workflow completion and scan coverage are not a guarantee that the project is free of vulnerabilities; review findings and test changes before deployment.
