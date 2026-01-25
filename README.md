# DPP - Final project

**Course:** Data Protection and Privacy    
**Paper:** *Preserving Privacy in Social Networks Against Neighborhood Attacks*      
**Register Number:** 5341007 

## Assumptions
1. Label generalization is omitted in the anonymization strategy to simplify the logic.
2. In `anonymize_graph`, candidate selection uses a heuristic based on degree for speed instead of computing the full anonymization cost between two vertices.

## Limitations
The custom implementation of the anonymization algorithm used in this work is not guaranteed to succeed in all cases.   
In particular, for certain input graphs and parameter settings, the algorithm may fail to produce a valid anonymized graph. Failure modes include:   
- returning a graph that is not isomorphic with respect to the equivalence classes induced during anonymization, and
- raising an `AnonymizationImpossibleError` when no suitable auxiliary node can be found outside the neighborhoods of the nodes being anonymized.

## File Structure

- **`analysis.ipynb`**  
  The main Jupyter Notebook reporting the analysis of the implemented anonymization algorithm. It includes examples, experiments, and visualizations of results.

- **`social_anonymizer.py`**  
  Implements the `SocialAnonymizer` class, which handles the anonymization process and acts as the main entry point for anonymizing graphs.

- **`label_domain.py`**  
  Implements the `LabelDomain` class, useful for label generalization. Note that in this project, label generalization is not applied, so this module is not actively used.

- **`mdfs_coder.py`**  
  Implements the `MDFSCoder` class, which generates the minimum DFS code of a graph to obtain a canonical representation, enabling custom graph isomorphism checking.

- **`utils.py`**  
  Provides utility functions for verifying isomprhism, such as checking if a graph satisfies k-anonymity, and other similar helper methods.

- **`metrics.py`**  
  Contains functions for computing global loss metrics and queryability scores to evaluate anonymization quality.

- **`visualization.py`**  
  Contains functions for visualizing graphs, degree distributions, and other data generated during anonymization.

- **`exceptions.py`**  
  Defines a custom exception used in `SocialAnonymizer` to handle a specific error scenario.
