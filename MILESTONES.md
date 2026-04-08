## **Task C7.T1 (CharmPheno)**

**Objective:** *Develop CharmPheno: an interpretable computational phenotyping capability for discovery and characterization of clinical phenotypes from patient health records, supporting patient-owned health insight*

**Task Description:** This task develops the **CharmPheno** computational phenotyping capability: an interpretable approach to discovering clinically meaningful patient phenotypes from structured patient health records, supporting per-patient phenotype profiles as a foundation for personalized health insight. The work develops a reusable, distributed inference framework that enables training on large-scale clinical data, applies it to discover latent clinical phenotypes, and characterizes those phenotypes in general and in specific populations (e.g. rare disease or pediatric oncology patients). Patient phenotype profiles are interpretable, probabilistic representations that serve as a substrate for a range of downstream capabilities, including phenotype-based patient similarity ("patients like me"), longitudinal profiles that evolve over time, and generation of plausible patient trajectories for risk exploration and what-if analysis. Trained phenotype models are suitable for deployment through the existing CHARMTwinsight model hosting infrastructure and for lightweight inference scenarios including on-device deployment (e.g. MyCharm) where a patient's own data need not leave their device.

**Location:** Work will be performed at UNC

**Primary Organization Responsible:** UNC

**Reports:** *Reporting on framework development, model performance, and phenotype characterization results will be provided as part of regular required reporting.*

**Human Subjects or Animal Research?:** No

### ***Sub-task C7.T1.3m***

**Objective:** *Design the architecture for the CharmPheno phenotype discovery capability and the distributed inference framework that supports it*

**Task Description:** Design the CharmPheno phenotype discovery approach and the computational framework that supports it. Specify modeling strategies for unsupervised discovery of clinical phenotypes from structured de-identified health records, identifying top candidate models (including Bayesian nonparametric approaches). Design the distributed inference framework needed to train such models on large-scale clinical datasets, including the base abstractions, data contracts, and training orchestration. The design will address patient privacy, reproducibility, and compatibility with the CHARMTwinsight model hosting service. The design document must include a security and privacy plan. This task aligns with the 27-month milestone (C7.Y1.M0.5).

**Location:** Work will be performed at UNC

**Primary Organization Responsible:** UNC

**Deliverables:** Deliverable C7.Y1.D3m-D: CharmPheno phenotype discovery strategy and supporting framework architecture design document, including security and privacy plan.

**Human Subjects or Animal Research?:** No

**Milestone C7.Y1.M0.5:** 27 months: Completion of CharmPheno phenotype discovery strategy and framework architecture design.

### ***Sub-task C7.T1.6m***

**Objective:** *Implement the CharmPheno phenotype discovery framework with initial phenotype model implementations, validated on synthetic data*

**Task Description:** Implement the CharmPheno computational phenotyping framework, including the model base class, training orchestration, diagnostics, and serialization. Test training established models on generated synthetic data from CHARMTwinsight, evaluating scalability, portability, and correctness against known latent structure. The resulting reusable, cross-platform framework will support a general class of phenotype models applicable to large-scale de-identified clinical data. This sub-task aligns with Deliverable C7.Y1.D6m at the 30-month milestone.

**Location:** Work will be performed at UNC

**Primary Organization Responsible:** UNC

**Deliverables:** Deliverable C7.Y1.D6m-CA: CharmPheno phenotype discovery framework with initial phenotype model implementations and synthetic validation results.

**Human Subjects or Animal Research?:** No

### ***Sub-task C7.T1.9m***

**Objective:** *Implement and explore additional CharmPheno phenotype discovery and characterization approaches; benchmark on synthetic and clinical datasets*

**Task Description:** Extend the CharmPheno framework with additional phenotype model implementations targeting richer phenotype discovery and characterization. Test candidate models on synthetic and de-identified clinical datasets and assess interpretability. This task aligns with the 33-month milestone (C7.Y1.M1.5).

**Location:** Work will be performed at UNC

**Primary Organization Responsible:** UNC

**Deliverables:** Deliverable C7.Y1.D9m-D: Report on the implementation and benchmarking of CharmPheno phenotype discovery and characterization approaches.

**Human Subjects or Animal Research?:** No

**Milestone C7.Y1.M1.5:** 33 months: Completion of implementation and benchmarking of additional CharmPheno phenotype discovery approaches.

### ***Sub-task C7.T1.12m***

**Objective:** *Apply CharmPheno to clinical data; deliver initial phenotype discovery and characterization results*

**Task Description:** Apply the CharmPheno phenotype discovery pipeline to clinical datasets at scale. Characterize discovered phenotypes (clusters of e.g. conditions, medications, procedures, etc.) by their composition using standard approaches, and their prevalence patterns across cohorts. Evaluate interpretability of the discovered phenotypes and identify directions for refinement. This sub-task aligns with Deliverable C7.Y1.D12m at the 36-month milestone.

**Location:** Work will be performed at UNC

**Primary Organization Responsible:** UNC

**Deliverables:** Deliverable C7.Y1.D12m-CA: CharmPheno phenotype discovery pipeline applied to clinical data, with initial phenotype discovery and characterization results.

**Human Subjects or Animal Research?:** No

---

## **Task C7.T2 (CharmPheno)**

**Objective:** *Integrate trained CharmPheno phenotype models with the CHARMTwinsight platform and develop patient-facing phenotype characterization capabilities*

**Task Description:** This task extends the CharmPheno computational phenotyping work into the CHARMTwinsight platform, focusing on phenotype model serving, patient-facing phenotype characterization capabilities, and interoperability. Trained phenotype models will be integrated with the existing CHARMTwinsight model hosting service to enable phenotype assignment and characterization for individual patient records. The task will explore phenotype characterization workflows, phenotype-trajectory visualization, and integration with other CHARM tools, including the mobile application MyCharm with local and hosted profiling.

**Location:** Work will be performed at UNC

**Primary Organization Responsible:** UNC

**Reports:** *Reporting on integration progress, patient-facing capability development, and platform hardening will be provided as part of regular required reporting.*

**Human Subjects or Animal Research?:** No

### ***Sub-task C7.T2.3m***

**Objective:** *Design the architecture for integrating trained CharmPheno phenotype models with CHARMTwinsight model hosting and patient-facing phenotype characterization capabilities*

**Task Description:** Design the integration architecture for deploying trained CharmPheno phenotype models through the CHARMTwinsight model hosting service. Specify interfaces for per-patient phenotype assignment, phenotype profile retrieval, and phenotype-trajectory characterization. Address model versioning, update workflows, and security considerations for serving phenotype models in patient-facing contexts, including on-device deployment. The design document must include a security and privacy plan. This task aligns with the 39-month milestone (C7.Y2.M0.5).

**Location:** Work will be performed at UNC

**Primary Organization Responsible:** UNC

**Deliverables:** Deliverable C7.Y2.D3m-D: CharmPheno integration and patient-facing phenotype characterization architecture design document, including security and privacy plan.

**Human Subjects or Animal Research?:** No

**Milestone C7.Y2.M0.5:** 39 months: Completion of CharmPheno integration and patient-facing capabilities architecture design.

### ***Sub-task C7.T2.6m***

**Objective:** *Develop exportable phenotype model pipeline and integrate CharmPheno with CHARMTwinsight model hosting for patient phenotype characterization*

**Task Description:** Implement the phenotype model export and deployment pipeline connecting the CharmPheno framework to the CHARMTwinsight model hosting service. Develop patient phenotype characterization capabilities accessible through the hosted model API, enabling per-patient phenotype assignment and phenotype profile retrieval for individual patient records. This sub-task aligns with Deliverable C7.Y2.D6m at the 42-month milestone.

**Location:** Work will be performed at UNC

**Primary Organization Responsible:** UNC

**Deliverables:** Deliverable C7.Y2.D6m-CA: CharmPheno phenotype model export pipeline and CHARMTwinsight hosting integration with patient phenotype characterization capabilities.

**Human Subjects or Animal Research?:** No

### ***Sub-task C7.T2.9m***

**Objective:** *Enhance phenotype characterization and phenotype-trajectory visualization capabilities; investigate FHIR-compatible inference workflows and on-device deployment feasibility*

**Task Description:** Enhance patient phenotype characterization workflows using served CharmPheno models, including visualization of per-patient phenotype profiles and phenotype-trajectory views over time. Investigate the feasibility of FHIR-compatible inference workflows, assessing the translation layer required to map FHIR clinical resources to the phenotype model's vocabulary at inference time. Assess lightweight inference runtimes suitable for on-device deployment, where trained phenotype models are shipped to patient devices and inference is performed locally against the patient's own data without transmitting it externally. This task aligns with the 45-month milestone (C7.Y2.M1.5).

**Location:** Work will be performed at UNC

**Primary Organization Responsible:** UNC

**Deliverables:** Deliverable C7.Y2.D9m-D: Report on phenotype characterization and visualization capabilities, including FHIR-compatible inference and on-device deployment feasibility assessment.

**Human Subjects or Animal Research?:** No

**Milestone C7.Y2.M1.5:** 45 months: Completion of phenotype characterization and visualization exploration, including FHIR compatibility and on-device deployment feasibility assessment.

### ***Sub-task C7.T2.12m***

**Objective:** *Deliver hardened CharmPheno phenotyping capabilities within CHARMTwinsight, with served phenotype models supporting patient phenotype characterization*

**Task Description:** Harden the CharmPheno platform integration to version readiness. Ensure robustness, ease of use, and documentation of the phenotype modeling components, the patient phenotype characterization capabilities, and any inference support developed during the exploration phase. This sub-task aligns with Deliverable C7.Y2.D12m at the 48-month milestone.

**Location:** Work will be performed at UNC

**Primary Organization Responsible:** UNC

**Deliverables:** Deliverable C7.Y2.D12m-CA: CHARMTwinsight with hardened CharmPheno phenotyping capabilities for patient phenotype characterization.

**Human Subjects or Animal Research?:** No
