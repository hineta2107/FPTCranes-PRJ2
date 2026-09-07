<!--
Sync Impact Report
- Version change: template (unversioned) -> 1.0.0
- Modified principles:
  - Template Principle 1 -> I. Data Integrity and Leakage Safety
  - Template Principle 2 -> II. Reproducible Staged Artifacts
  - Template Principle 3 -> III. Evidence-First Verification (NON-NEGOTIABLE)
  - Template Principle 4 -> IV. Offline Training, Read-Only Reporting
  - Template Principle 5 -> V. Honest Interpretation and Secure Defaults
- Added sections:
  - Scientific and Operational Constraints
  - Development Workflow and Quality Gates
- Removed sections: none
- Templates requiring updates:
  - ✅ .specify/templates/plan-template.md
  - ✅ .specify/templates/spec-template.md
  - ✅ .specify/templates/tasks-template.md
  - ✅ README.md
  - ✅ Command templates: directory not present; no stale references found
- Follow-up TODOs: none
-->
# AI Job Market Salary Prediction Constitution

## Core Principles

### I. Data Integrity and Leakage Safety

Every model change MUST declare the target, feature policy, development period, and locked
test period before target-aware analysis begins. Target values, target-derived fields,
identifiers, and post-outcome information MUST NOT enter model features. Preprocessors,
vocabularies, imputers, scalers, and feature selection MUST be fitted only on the applicable
training partition. Model selection and tuning MUST use development-period temporal
validation; the locked test MAY be opened only after selection is complete. These rules
protect reported performance from leakage and preserve the meaning of an unseen test.

### II. Reproducible Staged Artifacts

The documented 12-stage workflow MUST remain reproducible from the versioned source dataset
and pinned project dependencies. Stochastic operations MUST use an explicit seed. Each run
MUST emit inspectable intermediate evidence, final metrics, model metadata, and a serialized
inference bundle whose feature schema matches serving. Data provenance MUST include the input
path or identifier and a content hash. A change to an artifact name, schema, stage meaning, or
consumer contract MUST update every producer, consumer, test, and relevant document in the
same change.

### III. Evidence-First Verification (NON-NEGOTIABLE)

Behavior changes MUST begin with an executable test or validation criterion that fails for
the missing or incorrect behavior, followed by the smallest implementation that makes it
pass. Data and model changes MUST test leakage gates, split boundaries, schemas, deterministic
behavior, and metric calculations as applicable. Artifact or Streamlit changes MUST include
an integration test of the producer-consumer contract. Before merge, the relevant pytest
suite MUST pass and the documented pipeline and application entrypoints MUST be exercised in
proportion to the change. Printed values without assertions are not sufficient evidence.

### IV. Offline Training, Read-Only Reporting

Model fitting, tuning, and artifact generation MUST occur in the offline pipeline, never in
the Streamlit process. Streamlit MUST load the generated bundle, metadata, and report outputs
without mutating training data or model artifacts. Predictions MUST use the feature order,
categories, ranges, and vocabulary recorded by the bundle metadata. Missing, incompatible,
or corrupt artifacts MUST produce a clear actionable error rather than silent recomputation
or fabricated defaults. This boundary keeps the reporting layer fast, auditable, and
consistent with evaluated model behavior.

### V. Honest Interpretation and Secure Defaults

Reports and interfaces MUST distinguish descriptive correlation, fitted-model importance,
validation performance, locked-test performance, and causal claims. No result from this
dataset MAY be presented as causal salary economics or production fitness without external
validation. Prediction intervals MUST state their empirical basis and limitations. Demo
credentials MUST be identified as local-only; shared deployments MUST obtain credential
material from secrets or environment configuration and MUST NOT commit plaintext secrets.
User-controlled inputs and loaded artifact values MUST be validated before use.

## Scientific and Operational Constraints

- Python, the offline pipeline, generated file artifacts, and Streamlit are the established
  stack. New infrastructure requires a documented need and a simpler alternative analysis.
- The source dataset is an academic snapshot with contradictory and synthetic-looking
  structure. Results are suitable for supervised-regression study and controlled scenarios,
  not unqualified labor-market decisions.
- The five output packs, `artifacts/model_bundle.joblib`, and `artifacts/metadata.json` form a
  published internal contract. Compatibility changes MUST be explicit and tested.
- Generated evidence MUST retain units, partition names, model-selection basis, and enough
  provenance for a reviewer to trace a displayed claim back to its source artifact.
- Runtime failures MUST identify the missing or invalid input and the command needed to
  regenerate it. Silent fallbacks that alter scientific meaning are prohibited.

## Development Workflow and Quality Gates

1. Specifications MUST identify affected data boundaries, artifact contracts, scientific
   claims, security considerations, and measurable acceptance evidence.
2. Plans MUST pass every Constitution Check before research and again after design. Any
   exception MUST be recorded in Complexity Tracking with the need and rejected simpler
   alternative; exceptions cannot waive leakage safety or evidence requirements.
3. Tasks MUST put tests and validation before implementation, retain traceability to user
   stories, and include documentation and graph refresh work when interfaces or code change.
4. Reviews MUST inspect the actual diff, run relevant automated tests, and verify the exact
   pipeline or Streamlit entrypoint affected by the change. Model-result reviews MUST cite
   generated artifacts rather than recollection or console-only output.
5. Code changes MUST run `graphify update .` after verification so the repository knowledge
   graph remains synchronized. Documentation-only governance changes MAY skip the update when
   they do not alter indexed source relationships.

## Governance

This constitution supersedes conflicting project practices and generated template guidance.
An amendment MUST be proposed as a documented diff, state its motivation and migration impact,
update dependent templates and guidance in the same change, and receive maintainer approval.

Versions follow semantic versioning: MAJOR for incompatible principle removals or
redefinitions, MINOR for new principles or materially expanded obligations, and PATCH for
non-semantic clarification. The Sync Impact Report at the top of this file MUST record each
amendment and any deferred follow-up.

Every feature specification, implementation plan, task list, and code review MUST verify
constitutional compliance. Reviewers MUST reject unexplained violations. Complexity is
acceptable only when its necessity and simpler rejected alternative are documented. Runtime
commands and project-specific guidance remain in `README.md` and `AGENTS.md`, but neither may
override this constitution.

**Version**: 1.0.0 | **Ratified**: 2026-08-19 | **Last Amended**: 2026-08-19
