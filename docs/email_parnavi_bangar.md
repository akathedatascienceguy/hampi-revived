# Email — Collaboration Request v2

---

**To:** Parnavi Bangar
**From:** Nikita Gupta · Yashvardhan Gupta
**Subject:** Hampi Revived — A Collaboration Proposal

---

Dear Parnavi,

We hope this finds you well.

We are reaching out from super.money — Nikita as Head of Data Science, and Yash as a Founding Member — though this note is not about fintech. It is about Hampi.

Over the past several weeks, we have been building **Hampi Revived** — a computational pipeline that takes photographs of the Hampi World Heritage Site and attempts to do two things: reconstruct the surviving structures in 3D from multi-view photography, and use generative AI to predict what the damaged or partially destroyed monuments looked like when they were intact.

The project started as a technical exercise. It has become something we care about considerably more than that.

---

**What it does**

The pipeline runs in two tracks.

The first takes a set of overlapping photographs of a monument and reconstructs it as a navigable 3D point cloud and surface mesh — a kind of digital excavation without a trowel. The second, and more ambitious, track uses generative AI to complete what is missing. The model detects where a structure is ruined, masks that region, and uses the surviving stonework as context to fill in what is gone — the intact arch, the carved walls, the existing tiers all inform what gets generated. The lower portion of the image stays pixel-perfect original photography. Only the missing section is predicted.

We have run this on a partially destroyed entrance gopuram at Hampi. The model predicted a complete tiered shikhara tower rising above the original carved doorway, blended at the damage boundary. It is imperfect. But it holds up as a proof of concept — and it points clearly at what comes next.

---

**Where we are taking it — and why we are writing to you**

We are now fine-tuning the AI model specifically on Hampi. We have collated 50–100 photographs of Vijayanagara monuments from public archives and are training a LoRA — a technique that teaches the model the particular visual vocabulary of this site rather than a generic approximation of Indian temple architecture.

The difference is substantial. A model trained on Hampi learns the oxidised sandstone palette, the specific tier proportions of entrance gopurams in this complex, the density and rhythm of the carved friezes. Completions stop looking like plausible Dravidian architecture and start looking like *this place*.

But there is a hard ceiling to what photographs alone can teach.

Photographs cannot tell the model which structural elements are original versus later repair. They cannot convey the iconographic intent of a specific carved frieze, the period within the Vijayanagara chronology a structure belongs to, or which ruins are most archaeologically urgent. That knowledge lives in fieldwork and excavation records — in the kind of deep site familiarity that takes years to accumulate and cannot be scraped from the internet.

We believe you have that knowledge. And we think it is precisely what this pipeline needs to become genuinely useful rather than merely technically interesting.

---

**What we are proposing**

A collaboration, on whatever terms work for you. Concretely, we have three things in mind:

1. **Annotated images** — photographs with monument name, period, and structural condition would anchor the model's training to archaeologically grounded labels rather than visual guesswork
2. **Iconographic review** — your assessment of whether the AI-generated completions are plausible or wrong, so we can correct what the model has learned about what Vijayanagara structures actually looked like
3. **Priority monuments** — a short list of structures most in need of digital documentation, so our effort goes where it matters most

In return: full co-authorship on any resulting paper or exhibition, interactive 3D models of any monuments you specify, and the complete fine-tuned model and codebase — available to you under whatever terms you prefer.

We would be glad to share our current outputs and walk you through the pipeline at any time that suits you. Even a single conversation would move this meaningfully forward.

Thank you for your time and for your work on Hampi. The ruins deserve to be seen whole.

Warm regards,

**Nikita Gupta**
Head of Data Science, super.money

**Yashvardhan Gupta**
Founding Member in Data Science, super.money

---

*Project repository: github.com/akathedatascienceguy/hampi-revived*

---
> *"The ruins are not the absence of the empire — they are its most durable signature."*
