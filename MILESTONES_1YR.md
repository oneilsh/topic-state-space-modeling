## **Task C3.Tb**

**Objective:** *Develop and integrate distributed Bayesian state modeling capabilities for unsupervised clinical phenotype discovery, patient characterization, and simulation within CHARMTwinsight*

**Task Description:** This task designs, implements, and integrates a distributed Bayesian state modeling framework for unsupervised clinical phenotype discovery and patient characterization, with the goal of enabling Bayesian-based patient simulation and phenotyping within CHARMTwinsight. The work will design a general-purpose distributed variational inference framework suitable for large-scale clinical datasets, implement that framework as reusable infrastructure, train and evaluate Bayesian phenotype discovery models (Hierarchical Dirichlet Process and related approaches) on synthetic and clinical data, and integrate the resulting models with the CHARMTwinsight platform to support patient phenotyping and simulation. Because trained models consist of compact population-level parameters rather than patient data, the framework naturally supports lightweight deployment scenarios, including on-device inference where a patient's own data never leaves their device.

**Location:** Work will be performed at UNC

**Primary Organization Responsible:** UNC

**Reports:** *Reporting on framework development, model performance, phenotype characterization results, and platform integration will be provided as part of regular required reporting.*

**Human Subjects or Animal Research?:** No

### ***Sub-task C3.Tb.3m***

**Objective:** *Design the architecture for a distributed Bayesian state modeling framework and strategies for unsupervised clinical phenotype discovery from structured patient data*

**Task Description:** Design the computational framework for distributed variational inference on large-scale clinical datasets, including the base abstractions, data contracts, and training orchestration. Specify modeling strategies for unsupervised phenotype discovery from structured diagnosis code data, identifying candidate models (including the Hierarchical Dirichlet Process) and their suitability for clinical applications. The design will address patient privacy, reproducibility, and compatibility with the CHARMTwinsight model hosting service. The design document must include a security and privacy plan. This sub-task aligns with the 3-month milestone (C3b.M0.5).

**Location:** Work will be performed at UNC

**Primary Organization Responsible:** UNC

**Deliverables:** Deliverable C3b.D3m-D: Framework architecture and phenotype discovery strategy design document, including security and privacy plan.

**Human Subjects or Animal Research?:** No

**Milestone C3b.M0.5:** 3 months: Completion of framework architecture and phenotype discovery strategy design.

### ***Sub-task C3.Tb.6m***

**Objective:** *Implement the distributed variational inference framework as reusable infrastructure, validated on synthetic data*

**Task Description:** Implement the core framework, including the model base class, training orchestration, convergence diagnostics, and model serialization. Implement the distribute-and-aggregate training loop, convergence monitoring, and model export. Validate framework correctness on synthetic datasets with known latent structure. The framework is designed to be reusable: a model author defines the model-specific math, and the framework handles distribution across a Spark cluster, training loop management, convergence monitoring, and model export. This sub-task aligns with the 6-month milestone (C3b.M1) and with Deliverable C3b.D6m.

**Location:** Work will be performed at UNC

**Primary Organization Responsible:** UNC

**Deliverables:** Deliverable C3b.D6m-CA: Distributed variational inference framework implementation with synthetic validation results.

**Human Subjects or Animal Research?:** No

**Milestone C3b.M1:** 6 months: Completion of framework implementation and synthetic validation.

### ***Sub-task C3.Tb.9m***

**Objective:** *Train and evaluate Bayesian phenotype discovery models on clinical data using the implemented framework*

**Task Description:** Using the framework implemented in the prior sub-task, train Bayesian phenotype discovery models — principally the Hierarchical Dirichlet Process — on clinical datasets. Evaluate model quality, interpretability of discovered phenotypes, and computational performance. Characterize discovered phenotypes and their relationships to clinical outcomes. Iterate between model implementation refinements and training runs as needed: in practice, model implementation and training/evaluation are expected to be interleaved as evaluation results inform model and framework refinements. This sub-task aligns with the 9-month milestone (C3b.M1.5) and with Deliverable C3b.D9m.

**Location:** Work will be performed at UNC

**Primary Organization Responsible:** UNC

**Deliverables:** Deliverable C3b.D9m-D: Report on training and evaluation of Bayesian phenotype discovery models on clinical data, including phenotype characterization results, model quality assessment, and interpretability analysis.

**Human Subjects or Animal Research?:** No

**Milestone C3b.M1.5:** 9 months: Completion of model training and evaluation on clinical data.

### ***Sub-task C3.Tb.12m***

**Objective:** *Integrate trained Bayesian models with the CHARMTwinsight platform to support patient phenotyping and Bayesian-based patient simulation*

**Task Description:** Integrate trained Bayesian state models with the CHARMTwinsight platform through the existing model hosting service, enabling phenotype assignment and Bayesian-based patient simulation as platform capabilities. Develop patient phenotyping capabilities accessible through the hosted model API, enabling phenotype characterization for individual patient records. Develop patient simulation capabilities that leverage the trained models to generate posterior-predictive patient trajectories for use by downstream CHARMTwinsight tools. Address model versioning, update workflows, and security considerations for serving population-level models in patient-facing contexts. This sub-task aligns with the 12-month milestone (C3b.M2) and with Deliverable C3b.D12m.

**Location:** Work will be performed at UNC

**Primary Organization Responsible:** UNC

**Deliverables:** Deliverable C3b.D12m-CA: CHARMTwinsight integration of trained Bayesian state models, with hosted patient phenotyping and Bayesian-based patient simulation capabilities.

**Human Subjects or Animal Research?:** No

**Milestone C3b.M2:** 12 months: Completion of CHARMTwinsight integration with patient phenotyping and Bayesian-based patient simulation capabilities.
