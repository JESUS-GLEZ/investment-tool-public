# Third-party notices

This repository distributes project source. Dependencies are installed from their
upstream distributions and retain their own license and attribution terms.

- [Platypus-Opt](https://pypi.org/project/Platypus-Opt/1.4.1/), by David Hadka,
  provides multi-objective optimisation algorithms used by the portfolio engine.
  It is distributed separately under the GNU General Public License, version 3.
- The news sentiment feature downloads the
  [DistilBERT SST-2 model](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english/tree/714eb0fa89d2f80546fda750413ed43d93601a13)
  from Hugging Face when needed. The upstream model is licensed under Apache
  License 2.0. Its weights and training data are separate from this source release.

The Docker instructions build images locally. The dependency manifests and
lockfiles identify the Python and JavaScript dependencies used for those builds.
