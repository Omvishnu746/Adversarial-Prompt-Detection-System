# Adversarial Prompt Detection System

## Architectural Overview  

Introducing the complete architectural pivot of our system! 🚀  

### Tier-1: Fast Reject  
- Utilizes the SBERT semantic cache for sub-10ms matching, ensuring that only relevant inputs are processed swiftly. ⚡  

### Tier-2: Sliding-Window Inference  
- Implements DistilBERT for contextual analysis, processing inputs in a sliding-window fashion to capture nuanced adversarial patterns. 🔍  

### Tier-3: Decoupled Explainability  
- Features asynchronous background workers for explainability, allowing us to analyze model decisions without impacting throughput. 📊  

### Dataset  
- A stratified dataset comprising **12,032 rows**:  
  - **7,175 benign (60%)**  
  - **4,858 attacks (40%)**  
  - Perfectly balanced with a **1:1** ratio of jailbreaks vs prompt injections. 🗃️  
- Data in **JSON schema format** for easy consumption and integration.  

### Updated Models Table  
- Currently only supporting:  
  - **DistilBERT**  
  - **SBERT**  

### Architecture Diagram  
- A sequential **5-tier pipeline architecture diagram** illustrating the flow of data and decision logic. 🏗️  

### Updated Data Sources  
- Integrates data from diverse sources like:  
  - databricks-dolly-15k  
  - JailbreakBench  
  - lmsys/toxic-chat  
  - PKU-SafeRLHF-QA  
  - deepset  
  - neuralchemy  
  - wambosec  

### Performance Targets  
- Target latency of **<50ms** for rapid adversarial detection. ⏱️  

Maintain professional markdown formatting throughout the document. 💼  
