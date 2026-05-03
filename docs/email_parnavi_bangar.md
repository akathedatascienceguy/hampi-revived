# Email — Collaboration Request v2

---

**To:** Parnavi Bangar
**From:** Nikita Gupta · Yashvardhan Gupta
**Subject:** Hampi Revived — A Collaboration Proposal

---

Dear Parnavi,

We hope this finds you well.

We are reaching out from super.money — Nikita as Head of Data Science, and Yash as a Founding Member — though this note is not about fintech. It is about Hampi, and about *Reminiscing History*.

We came across your publication recently and read it with genuine admiration. The way it brings historical sites and their stories to a general audience — without sacrificing depth — is exactly the register we have been trying to reach with the project we are about to describe. We believe there may be a natural overlap, and we will come to that at the end of this note.

Over the past several weeks, we have been building **Hampi Revived** — a computational pipeline that takes photographs of the Hampi World Heritage Site and attempts to do two things: reconstruct the surviving structures in 3D from multi-view photography, and use generative AI to predict what the damaged or partially destroyed monuments looked like when they were intact.

The project started as a technical exercise. It has become something we care about considerably more than that.

---

**You can see it live here:**
**[https://hampi-revived.streamlit.app](https://hampi-revived.streamlit.app)**

The app walks through the full pipeline — raw photographs, 3D point cloud, and the AI-generated restorations — and includes a section on the history of the Vijayanagara Empire and the architectural vocabulary of Hampi. We would be glad if you took five minutes to look at it.

---

**What it does**

The pipeline runs in two tracks.

The first takes a set of overlapping photographs of a monument and reconstructs it as a navigable 3D point cloud and surface mesh — a kind of digital excavation without a trowel. The second, and more ambitious, track uses generative AI to complete what is missing. The model detects where a structure is ruined, masks that region, and uses the surviving stonework as context to fill in what is gone — the intact arch, the carved walls, the existing tiers all inform what gets generated. The lower portion of the image stays pixel-perfect original photography. Only the missing section is predicted.

We have run this on partially destroyed gopurams at Hampi — the entrance gateway to the Vijayanagara complex at Hazara Rama, the truncated north gopuram, and several others. The model predicts a complete tiered shikhara tower rising above the original carved doorway, blended at the damage boundary.

---

**Where we are technically**

The pipeline has gone through several iterations. The current version:

- Trains a **rank-8 LoRA** (low-rank adaptation) on 70+ Hampi and South Indian temple photographs over 600 steps — teaching the model the specific visual vocabulary of Vijayanagara architecture rather than a generic approximation
- Uses a **sky-contour damage mask** that follows the actual silhouette of each structure rather than cutting at a flat horizontal line
- Applies **ControlNet-Canny** conditioning so that the generated upper tiers are geometrically anchored to the surviving lower structure — pillar alignment, arch width, and stone rhythm are preserved
- Conditions generation on a **reference photograph of an intact gopuram** (Virupaksha Temple, same complex) via IP-Adapter, so the proportions of the generated shikhara draw from a real complete example on the same site
- Uses **synthetic damage pairs** during training: intact gopuram images are programmatically masked and the model is trained to reconstruct the masked-out upper portion from the intact lower context — directly teaching the restoration task rather than just architectural style
- Applies **LAB colour matching** after compositing to ensure the generated region's colour temperature matches the original stonework

The results are meaningfully better than the baseline. The stone colour, carving density, and tower profile are recognisably Hampi rather than generically Dravidian.

The remaining hard ceiling is data: photographs cannot convey iconographic intent, period attribution within the Vijayanagara chronology, or which structural elements are original versus later repair. That knowledge lives in fieldwork.

---

**Where we are taking it — and why we are writing to you**

We believe you have that knowledge. And we think it is precisely what this pipeline needs to become genuinely useful rather than merely technically interesting.

The difference between a model that has seen photographs and one that has been annotated by someone who has worked the Hampi site is the difference between plausible and accurate. We want to build toward the latter.

---

**A thought about *Reminiscing History***

When we read your publication, one thing struck us: the most affecting pieces are those where the reader can *see* what once existed alongside what remains. Our reconstructions — imperfect as they currently are — do exactly that. A photograph of a ruined gopuram beside its computationally completed version communicates the scale of what was lost in a way that prose alone cannot.

We think these reconstructions could find a natural home in a future edition of *Reminiscing History* — particularly a piece on Hampi, or on computational approaches to heritage documentation more broadly. If that is of interest, we would be glad to produce publication-quality renders of any monument you choose, at whatever resolution the layout requires, with full attribution and editorial control on your side.

---

**What we are proposing**

A collaboration, on whatever terms work for you. Concretely, we have three things in mind:

1. **Annotated images** — photographs with monument name, period, and structural condition would anchor the model's training to archaeologically grounded labels rather than visual guesswork
2. **Iconographic review** — your assessment of whether the AI-generated completions are plausible or wrong, so we can correct what the model has learned about what Vijayanagara structures actually looked like
3. **Priority monuments** — a short list of structures most in need of digital documentation, so our effort goes where it matters most

In return: full co-authorship on any resulting paper or exhibition, interactive 3D models of any monuments you specify, publication-quality digital reconstructions for use in *Reminiscing History* or any other platform you see fit, and the complete fine-tuned model and codebase — available to you under whatever terms you prefer.

We would be glad to share our current outputs and walk you through the pipeline at any time that suits you. The live app at **[hampi-revived.streamlit.app](https://hampi-revived.streamlit.app)** gives a quick overview; a call would let us go deeper into the methodology and show you the per-monument restoration results.

Even a single conversation would move this meaningfully forward.

Thank you for your time and for your work on Hampi. The ruins deserve to be seen whole.

Warm regards,

**Nikita Gupta**
Head of Data Science, super.money

**Yashvardhan Gupta**
Founding Member in Data Science, super.money

---

*Live app: [https://hampi-revived.streamlit.app](https://hampi-revived.streamlit.app)*
*Repository: [github.com/akathedatascienceguy/hampi-revived](https://github.com/akathedatascienceguy/hampi-revived)*

---

> *"The ruins are not the absence of the empire — they are its most durable signature."*
