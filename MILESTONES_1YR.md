## **Task C7.T1 (CharmPheno)**

**Objective:** *Develop and integrate CharmPheno: an interpretable computational phenotyping capability for discovery and characterization of clinical phenotypes from patient health records, with hosted integration into the CHARMTwinsight platform*

**Task Description:**
This task develops the **CharmPheno** computational phenotyping capability:
an interpretable approach to discovering interpretable and clinically meaningful patient *phenotype profiles* from structured health records as a foundation for personalized health insight.
The work develops a reusable, distributed-compute inference framework that enables training on large-scale clinical data,
and applies it to discover latent phenotypes contributing to profiles.
It also characterizes the resulting phenotypes and profiles in general, in specific populations (rare disease and pediatric oncology patients), and at the individual level.
Patient phenotype profiles are interpretable, probabilistic representations that serve a range of downstream capabilities,
including profile-based patient matching ("patients-like-me"),
profiles that evolve over time,
and autoregressive generation of patient profiles for risk exploration and what-if analyses.
Trained profile-generating models do not contain sensitive information, and are integrated with the existing CHARMTwinsight model hosting infrastructure to support per-patient phenotype characterization through the platform.

**Location:** Work will be performed at UNC

**Primary Organization Responsible:** UNC

**Reports:** *Reporting on framework development, model performance, phenotype characterization results, and platform integration will be provided as part of regular required reporting.*

**Human Subjects or Animal Research?:** No

### ***Sub-task C7.T1.3m***

**Objective:** *Design the architecture for the CharmPheno phenotype discovery capability and the distributed inference framework that supports it*

**Task Description:** Design the CharmPheno phenotype discovery approach and the computational framework that supports it. Specify modeling strategies for unsupervised discovery of clinical phenotypes from structured de-identified health records, identifying top candidate models (including Bayesian nonparametric approaches). Design the distributed inference framework needed to train such models on large-scale clinical datasets, including the base abstractions, data contracts, and training orchestration. The design will address patient privacy, reproducibility, and compatibility with the CHARMTwinsight model hosting service. The design document must include a security and privacy plan. This sub-task aligns with the 3-month milestone (C7.Y1.M0.5).

**Location:** Work will be performed at UNC

**Primary Organization Responsible:** UNC

**Deliverables:** Deliverable C7.Y1.D3m-D: CharmPheno phenotype discovery strategy and supporting framework architecture design document, including security and privacy plan.

**Human Subjects or Animal Research?:** No

**Milestone C7.Y1.M0.5:** 3 months: Completion of CharmPheno phenotype discovery strategy and framework architecture design.

### ***Sub-task C7.T1.6m***

**Objective:** *Implement the CharmPheno phenotype discovery framework as reusable infrastructure, validated on synthetic data*

**Task Description:** Implement the CharmPheno computational phenotyping framework, including the model base class, training orchestration, diagnostics, and serialization. Test training established models on generated synthetic data from CHARMTwinsight and de-identified clinical data, evaluating scalability, portability, and correctness. The resulting reusable, cross-platofrm framework will support a general class of phenotype models applicable to large-scale de-identified clinical data. This sub-task aligns with the 6-month milestone (C7.Y1.M1) and with Deliverable C7.Y1.D6m.

**Location:** Work will be performed at UNC

**Primary Organization Responsible:** UNC

**Deliverables:** Deliverable C7.Y1.D6m-CA: CharmPheno phenotype discovery framework implementation with synthetic validation results.

**Human Subjects or Animal Research?:** No

**Milestone C7.Y1.M1:** 6 months: Completion of CharmPheno framework implementation and synthetic validation.

### ***Sub-task C7.T1.9m***

**Objective:** *Train and evaluate CharmPheno phenotype discovery models on clinical data using the implemented framework*

**Task Description:** Apply the CharmPheno phenotype discovery framework to clinical datasets. Test candidate models on de-identified clinical data and assess interpretability. Characterize discovered phenotypes (clusters of e.g. conditions, medications, procedures, etc.) by their composition using standard approaches, and their prevalence patterns across cohorts. Iterate between model implementation refinements and training runs as needed. This sub-task aligns with the 9-month milestone (C7.Y1.M1.5) and with Deliverable C7.Y1.D9m.

**Location:** Work will be performed at UNC

**Primary Organization Responsible:** UNC

**Deliverables:** Deliverable C7.Y1.D9m-D: Report on training and evaluation of CharmPheno phenotype discovery models on clinical data, including phenotype characterization results, model quality assessment, and interpretability analysis.

**Human Subjects or Animal Research?:** No

**Milestone C7.Y1.M1.5:** 9 months: Completion of CharmPheno phenotype model training and evaluation on clinical data.

### ***Sub-task C7.T1.12m***

**Objective:** *Integrate trained CharmPheno phenotype models with the CHARMTwinsight platform to support per-patient phenotype characterization and phenotype-trajectory visualization*

**Task Description:** Implement the phenotype model export and deployment pipeline connecting the CharmPheno framework to the CHARMTwinsight model hosting service. Develop patient phenotype characterization capabilities accessible through the hosted model API, enabling per-patient phenotype assignment, phenotype profile retrieval, and phenotype-trajectory visualization for use by downstream CHARMTwinsight tools. Address model versioning, update workflows, and security considerations for serving phenotype models in patient-facing contexts. This sub-task aligns with the 12-month milestone (C7.Y1.M2) and with Deliverable C7.Y1.D12m.

**Location:** Work will be performed at UNC

**Primary Organization Responsible:** UNC

**Deliverables:** Deliverable C7.Y1.D12m-CA: CHARMTwinsight integration of trained CharmPheno phenotype models, with hosted per-patient phenotype characterization and phenotype-trajectory visualization capabilities.

**Human Subjects or Animal Research?:** No

**Milestone C7.Y1.M2:** 12 months: Completion of CHARMTwinsight integration with CharmPheno per-patient phenotype characterization capabilities.
