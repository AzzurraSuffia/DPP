# DPP - Final project

**Course:** Data Protection and Privacy    
**Paper:** *Preserving Privacy in Social Networks Against Neighborhood Attacks*      
**Register Number:** 5341007 

## Assumptions
1. Label generalization is omitted in the anonymization strategy to simplify the logic.
2. In `anonymize_graph`, candidate selection uses a heuristic based on degree for speed instead of computing the full anonymization cost between two vertices.

## File Structure

- **`analysis.ipynb`**  
  The main Jupyter Notebook reporting the analysis of the implemented anonymization algorithm. It includes examples, experiments, and visualizations of results.

- **`social_anonymizer.py`**  
  Implements the `SocialAnonymizer` class, which handles the anonymization process and acts as the main entry point for anonymizing graphs.

- **`label_domain.py`**  
  Implements the `LabelDomain` class, useful for label generalization. Note that in this project, label generalization is not applied, so this module is not actively used.

- **`mdfs_coder.py`**  
  Implements the `MDFSCoder` class, which generates the minimum DFS code of a graph to obtain a canonical representation, enabling efficient graph isomorphism checking.

- **`utils.py`**  
  Provides utility functions for verifying graph properties, such as checking if a graph satisfies k-anonymity, and other helper methods.

- **`metrics.py`**  
  Contains functions for computing global loss metrics and queryability scores to evaluate anonymization quality.

- **`visualization.py`**  
  Contains functions for visualizing graphs, degree distributions, and other data generated during anonymization.

- **`exceptions.py`**  
  Defines custom exceptions used in `SocialAnonymizer` to handle specific error scenarios, such as impossible anonymization.
