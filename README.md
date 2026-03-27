# Bayesian State Modeling for Clinical Phenotype Discovery and Patient Outcomes

This project develops a distributed Bayesian state modeling framework for
unsupervised clinical phenotype discovery and outcome characterization, with the
goal of enabling personalized, patient-facing health insights.

```mermaid
graph LR
    A[Millions of<br/>Clinical Visits] --> B[Unsupervised<br/>Phenotype Discovery]
    B --> C[Patient State<br/>Dynamics]
    C --> D[Outcome<br/>Characterization]
    D --> E[Patient-Facing<br/>Insights]

    style A fill:#f9f,stroke:#333
    style E fill:#bfb,stroke:#333
```

The work spans three layers: a research design for the clinical modeling approach,
a general-purpose software framework for distributed variational inference, and a
milestone plan for delivering these capabilities within the CHARMTwinsight platform.

---

## How It Works

Clinical visits are bags of diagnosis codes. The framework discovers latent
phenotypes — recurring patterns of co-occurring diagnoses — then models how
individual patients move through those phenotypes over time.

```mermaid
graph TD
    subgraph "Stage 1: Phenotype Discovery"
        V1["Visit: E11.9, I10,<br/>E78.5, Z79.84"] --> T1["Metabolic<br/>Syndrome"]
        V2["Visit: J44.1,<br/>J96.0, R06.0"] --> T2["Respiratory<br/>Failure"]
        V3["Visit: F32.1,<br/>G47.0, R45.8"] --> T3["Depression /<br/>Sleep"]
    end

    subgraph "Stage 2: Patient Dynamics"
        T1 & T2 & T3 --> TS["Patient Phenotype<br/>Trajectory Over Time"]
        TS --> IG["Interaction Graph:<br/>Which phenotypes<br/>drive others?"]
        TS --> PR["Outcome Prediction:<br/>Where is this patient<br/>heading?"]
    end

    style T1 fill:#ffd,stroke:#333
    style T2 fill:#ffd,stroke:#333
    style T3 fill:#ffd,stroke:#333
    style IG fill:#bfb,stroke:#333
    style PR fill:#bfb,stroke:#333
```

## Distributed Training, Compact Models

The framework distributes computation across Spark workers using a
broadcast→update→aggregate pattern. Trained models are compact population-level
parameters (~30-60MB) containing no patient data — small enough to ship to a
patient's phone for private, on-device inference.

```mermaid
graph LR
    subgraph "Training (Spark Cluster)"
        direction TB
        G["Global Parameters"] -->|broadcast| W1["Worker 1<br/>local update"]
        G -->|broadcast| W2["Worker 2<br/>local update"]
        G -->|broadcast| W3["Worker N<br/>local update"]
        W1 -->|stats| AG["Aggregate &<br/>Update"]
        W2 -->|stats| AG
        W3 -->|stats| AG
        AG --> G
    end

    subgraph "Deployment"
        direction TB
        EX["Export<br/>JSON + .npy"] --> MH["Model Hosting<br/>Service"]
        EX --> PH["Patient Device<br/>On-Device Inference"]
    end

    AG --> EX

    style G fill:#ffd,stroke:#333
    style MH fill:#bfb,stroke:#333
    style PH fill:#bfb,stroke:#333
```

---

## Documents

- **[Topic-State Modeling Research Design](TOPIC_STATE_MODELING.md)** — The scientific
  foundation. Describes a two-stage approach: discovering clinical phenotypes from
  diagnosis code data using a Hierarchical Dirichlet Process, then modeling patient
  dynamics through those phenotypes using a sparse Ornstein-Uhlenbeck process.
  Includes background, model architecture, computational design, and references.

- **[spark-vi Framework Design](SPARK_VI_FRAMEWORK.md)** — The software architecture.
  A PySpark-native framework for distributed variational inference where model authors
  implement the math and the framework handles Spark orchestration, training loops,
  diagnostics, and model export. Notebook-first, with compact privacy-friendly model
  artifacts suitable for lightweight deployment including on-device inference.

- **[Milestones (C3.T3b / C3.T4b)](MILESTONES.md)** — The delivery plan. Eight
  quarterly milestones across two years: Year 3 builds the framework and applies it
  to clinical data; Year 4 integrates trained models with CHARMTwinsight model hosting
  and explores patient-facing outcome capabilities.
