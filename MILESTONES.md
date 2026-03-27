## **Task C3.T3b**

**Objective:** *Develop Bayesian state modeling capabilities for unsupervised clinical phenotype discovery and outcome characterization within CHARMTwinsight*

**Task Description:** This task develops a distributed Bayesian state modeling framework and applies it to structured clinical data for unsupervised phenotype discovery and outcome characterization, with the goal of enabling personalized, patient-facing health insights. The work will design and implement a general-purpose distributed variational inference framework suitable for large-scale clinical datasets, implement and explore Bayesian models for discovering latent clinical phenotypes from diagnosis code data, and benchmark approaches on synthetic and reference datasets before applying them to de-identified clinical data. The framework will be designed for compatibility with the existing CHARMTwinsight model hosting infrastructure. Because trained models consist of compact population-level parameters rather than patient data, the framework naturally supports lightweight deployment scenarios — including on-device inference where a patient's own data never leaves their device.

**Location:** Work will be performed at UNC

**Primary Organization Responsible:** UNC

**Reports:** *Reporting on framework development, model performance, and phenotype characterization results will be provided as part of regular required reporting.*

**Human Subjects or Animal Research?:** No

### ***Sub-task C3.T3b.3m***

**Objective:** *Design the architecture for a distributed Bayesian state modeling framework and strategies for unsupervised clinical phenotype discovery from structured patient data*

**Task Description:** Design the computational framework for distributed variational inference on large-scale clinical datasets, including the base abstractions, data contracts, and training orchestration. Specify modeling strategies for unsupervised phenotype discovery from structured diagnosis code data, identifying candidate models and their suitability for clinical applications. The design will address patient privacy, reproducibility, and compatibility with the CHARMTwinsight model hosting service. The design document must include a security and privacy plan. This task aligns with the 27-month milestone (C3b.Y3.M0.5).

**Location:** Work will be performed at UNC

**Primary Organization Responsible:** UNC

**Deliverables:** Deliverable C3b.Y3.D3m-D: Framework architecture and phenotype discovery strategy design document, including security and privacy plan.

**Human Subjects or Animal Research?:** No

**Milestone C3b.Y3.M0.5:** 27 months: Completion of framework architecture and phenotype discovery strategy design.

### ***Sub-task C3.T3b.6m***

**Objective:** *Implement the distributed variational inference framework with initial model implementations, validated on synthetic data*

**Task Description:** Implement the core framework, including the model base class, training orchestration, convergence diagnostics, and model serialization. Develop initial model implementations targeting phenotype discovery from discrete clinical data. Validate correctness and parameter recovery using synthetic datasets with known latent structure. This sub-task aligns with Deliverable C3b.Y3.D6m at the 30-month milestone.

**Location:** Work will be performed at UNC

**Primary Organization Responsible:** UNC

**Deliverables:** Deliverable C3b.Y3.D6m-CA: Distributed variational inference framework with initial model implementations and synthetic validation results.

**Human Subjects or Animal Research?:** No

### ***Sub-task C3.T3b.9m***

**Objective:** *Implement and explore additional modeling approaches for phenotype characterization; benchmark on synthetic and reference datasets*

**Task Description:** Extend the framework with additional model implementations targeting phenotype characterization and outcome modeling. Benchmark candidate approaches on synthetic and, where available, de-identified reference datasets. Assess model quality, interpretability of discovered phenotypes, computational performance, and scalability. This task aligns with the 33-month milestone (C3b.Y3.M1.5).

**Location:** Work will be performed at UNC

**Primary Organization Responsible:** UNC

**Deliverables:** Deliverable C3b.Y3.D9m-D: Report on the implementation and benchmarking of modeling approaches for clinical phenotype characterization.

**Human Subjects or Animal Research?:** No

**Milestone C3b.Y3.M1.5:** 33 months: Completion of implementation and benchmarking of additional modeling approaches.

### ***Sub-task C3.T3b.12m***

**Objective:** *Apply the modeling pipeline to clinical data; deliver initial phenotype discovery and outcome characterization results*

**Task Description:** Apply the developed pipeline to clinical datasets. Characterize discovered phenotypes and their relationships to clinical outcomes. Evaluate clinical interpretability and identify directions for refinement. This sub-task aligns with Deliverable C3b.Y3.D12m at the 36-month milestone.

**Location:** Work will be performed at UNC

**Primary Organization Responsible:** UNC

**Deliverables:** Deliverable C3b.Y3.D12m-CA: Modeling framework applied to clinical data, with initial phenotype discovery and outcome characterization results.

**Human Subjects or Animal Research?:** No

---

## **Task C3.T4b**

**Objective:** *Integrate trained Bayesian state models with CHARMTwinsight model hosting and develop patient-facing outcome capabilities*

**Task Description:** This task extends the Bayesian state modeling work into the CHARMTwinsight platform, focusing on model serving, patient-facing capabilities, and interoperability. Trained models will be integrated with the existing model hosting service to enable phenotype-based patient characterization. The task will explore outcome prediction, simulation, and visualization capabilities, integrate with other CHARM tools, and assess the feasibility of lightweight on-device deployment for privacy-preserving patient-facing applications.

**Location:** Work will be performed at UNC

**Primary Organization Responsible:** UNC

**Reports:** *Reporting on integration progress, patient-facing capability development, and platform hardening will be provided as part of regular required reporting.*

**Human Subjects or Animal Research?:** No

### ***Sub-task C3.T4b.3m***

**Objective:** *Design the architecture for integrating trained models with CHARMTwinsight model hosting and patient-facing outcome capabilities*

**Task Description:** Design the integration architecture for deploying trained Bayesian state models through the CHARMTwinsight model hosting service and potentially directly to patient devices. Specify interfaces for patient characterization, outcome prediction, and model-driven simulation. Address model versioning, update workflows, and security considerations for serving population-level models in patient-facing contexts, including an assessment of on-device deployment where patient data remains local. The design document must include a security and privacy plan. This task aligns with the 39-month milestone (C3b.Y4.M0.5).

**Location:** Work will be performed at UNC

**Primary Organization Responsible:** UNC

**Deliverables:** Deliverable C3b.Y4.D3m-D: Model integration and patient-facing outcome capabilities architecture design document, including security and privacy plan.

**Human Subjects or Animal Research?:** No

**Milestone C3b.Y4.M0.5:** 39 months: Completion of model integration and patient-facing capabilities architecture design.

### ***Sub-task C3.T4b.6m***

**Objective:** *Develop exportable model pipeline and integrate with CHARMTwinsight model hosting for patient characterization*

**Task Description:** Implement the model export and deployment pipeline connecting the modeling framework to the CHARMTwinsight model hosting service. Develop patient characterization capabilities accessible through the hosted model API, enabling phenotype assignment and outcome estimation for individual patient records. This sub-task aligns with Deliverable C3b.Y4.D6m at the 42-month milestone.

**Location:** Work will be performed at UNC

**Primary Organization Responsible:** UNC

**Deliverables:** Deliverable C3b.Y4.D6m-CA: Model export pipeline and hosting integration with patient characterization capabilities.

**Human Subjects or Animal Research?:** No

### ***Sub-task C3.T4b.9m***

**Objective:** *Explore outcome prediction, simulation, and visualization capabilities; investigate FHIR-compatible inference workflows*

**Task Description:** Explore outcome prediction and simulation capabilities using served models, including visualization of phenotype characterizations and outcome estimates. Investigate the feasibility of FHIR-compatible inference workflows, assessing the translation layer required to map FHIR clinical resources to model vocabularies at inference time. Assess lightweight inference runtimes suitable for on-device deployment, where trained population-level models are shipped to patient devices and inference is performed locally against the patient's own data without transmitting it externally. This task aligns with the 45-month milestone (C3b.Y4.M1.5).

**Location:** Work will be performed at UNC

**Primary Organization Responsible:** UNC

**Deliverables:** Deliverable C3b.Y4.D9m-D: Report on outcome prediction and simulation capabilities, including FHIR-compatible inference and on-device deployment feasibility assessment.

**Human Subjects or Animal Research?:** No

**Milestone C3b.Y4.M1.5:** 45 months: Completion of outcome and simulation exploration, including FHIR compatibility and on-device deployment assessment.

### ***Sub-task C3.T4b.12m***

**Objective:** *Deliver hardened Bayesian state modeling capabilities within CHARMTwinsight, with served models supporting patient characterization and outcome exploration*

**Task Description:** Harden the platform integration to version readiness. Ensure robustness, scalability, and documentation of the model serving pipeline, patient characterization capabilities, and any FHIR-compatible inference support developed during the exploration phase. This sub-task aligns with Deliverable C3b.Y4.D12m at the 48-month milestone.

**Location:** Work will be performed at UNC

**Primary Organization Responsible:** UNC

**Deliverables:** Deliverable C3b.Y4.D12m-CA: CHARMTwinsight with hardened Bayesian state modeling capabilities for patient characterization and outcome exploration.

**Human Subjects or Animal Research?:** No
