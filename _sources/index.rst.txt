Archai Documentation
====================

**Archai** is a Neural Architecture Search (NAS) framework built upon PyTorch. It provides a comprehensive solution for automating the process of finding the optimal architecture for deep learning models, making it easier for researchers and practitioners to achieve state-of-the-art results. First launched as an open-source project in 2020, Archai has made impactful progress by forming a positive feedback loop between the engineering and research aspects.

It has innovated on both search algorithms and search spaces, explored ideas on zero-cost proxies of architecture accuracy and in very recent work explored novel more efficient alternatives to the ubiquitious attention operator which is now informing next-generation search-space design. Additionally, it offers the following advantages:

* 🔬 Easy mix-and-match between different algorithms;

* 📈 Self-documented hyper-parameters and fair comparison;

* ⚡ Extensible and modular to allow rapid experimentation;

* 📂 Powerful configuration system and easy-to-use tools.


Citing Archai
-------------

If you use Archai in a scientific publication, please consider citing it:

.. code-block:: latex

   @misc{Archai:22,
      title=Archai: Platform for Neural Architecture Search,
      url=https://www.microsoft.com/en-us/research/project/archai-platform-for-neural-architecture-search,
      journal=Microsoft Research,
      year=2022,
      month=Jul
   }

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Getting Started

   Installation <getting_started/installation>
   Package Structure <getting_started/package_structure>
   Quick Start <getting_started/quick_start>
   Notebooks <getting_started/notebooks>

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Advanced Guide

   Neural Architecture Search <advanced_guide/nas>
   Computer Vision <advanced_guide/cv>
   Natural Language Processing <advanced_guide/nlp>
   Cloud-Based Search <advanced_guide/cloud>

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Contributing

   First Time Contributor <contributing/first_contribution>
   Documentation <contributing/documentation>
   Unit Tests <contributing/unit_tests>

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Support

   Frequently Asked Questions <support/faq>
   Contact <support/contact>
   Copyright <support/copyright>

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Reference

   API <reference/api>
   Roadmap <reference/roadmap>
   Changelog <reference/changelog>
