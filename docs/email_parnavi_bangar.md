# Email — Collaboration Request: Hampi Revived

---

**To:** Parnavi Bangar
**From:** Nikita Gupta · Yashvardhan Gupta
**Subject:** Bringing Hampi's Ruins Back — A Data Science Project, and a Conversation

---

Dear Parnavi,

We hope this finds you well.

We are reaching out from super.money — Nikita as Head of Data Science, and Yash as a Founding Member — though this note is not about fintech. It is about Hampi.

Over the past several weeks, we have been building **Hampi Revived** — a computational pipeline that takes photographs of the Hampi World Heritage Site and attempts to do two things: reconstruct the surviving structures in 3D from multi-view photography, and use generative AI to predict what the damaged or partially destroyed monuments looked like when they were intact.

The project started as a technical exercise. It has become something we care about considerably more than that.

---

**What we have built so far**

The pipeline runs end-to-end and costs nothing to run — entirely on open-source tools and free cloud APIs:

| Stage | What happens |
|---|---|
| Preprocessing | CLAHE contrast enhancement + denoising, optimised for carved-granite photography |
| Feature extraction | SIFT keypoint detection (4,100+ features per image), FLANN matching across image pairs |
| 3D reconstruction | Structure from Motion → 106,000+ point dense cloud → Poisson surface mesh |
| Generative completion | SDXL inpainting with a damage mask: the model fills *only* the missing section of a ruined monument using the surviving stonework as context |

The most recent result is a restoration of a partially destroyed entrance gopuram. The model identified the damage boundary automatically (the row where Laplacian sharpness drops sharply), masked the ruined top, and predicted a complete tiered shikhara tower rising above the existing intact carved doorway. The lower 59% of the image — the ornate arch, flanking walls, the stone path — is untouched, pixel-perfect from the original photograph.

It is imperfect. The generated tower is plausible Dravidian architecture but not specifically Vijayanagara — the model does not yet know the particular vocabulary of this site. That is precisely the gap we are writing to you about.

---

**Why we are reaching out to you**

The single most impactful improvement we could make to this pipeline is training it on Hampi-specific data. Not generic "Indian temple" imagery, but annotated photographs of Vijayanagara structures: the specific sandstone colour, the tier proportions of entrance gopurams in this complex, the density and style of the carved friezes, the relationship between pillar height and superstructure across different monument types.

Your work on Hampi — the research, the documentation, the deep familiarity with the site — represents exactly the kind of grounded knowledge that our model currently lacks and cannot learn from the internet alone.

We would love to explore whether there is a form of collaboration that works for you. We are thinking about:

1. **Annotated image datasets** — site-specific photographs with monument labels, period, and structural condition, which would train a fine-tuned model anchored to Vijayanagara architectural vocabulary
2. **Iconographic validation** — a layer where the AI-generated completions are reviewed and corrected against your knowledge, producing a historically defensible output rather than a visually plausible one
3. **Conservation mapping** — overlaying the 3D mesh against known structural vulnerabilities to produce a damage-priority heatmap that ASI conservators could use practically

In return: all code is open-sourced and fully available to you, we would offer full co-authorship on any resulting paper or exhibition, and we can generate interactive 3D models of any specific monuments you specify — suitable for digital archival, publication, or exhibition use.

---

**In short**

We have built a technically functional pipeline. What it needs to become genuinely useful — archaeologically, conservationally, historically — is the kind of domain knowledge that takes years to accumulate and cannot be scraped from the web. We think you might have that knowledge, and we would be glad to find out if this is a conversation worth having.

We are happy to share the current outputs, the full codebase, and a live walkthrough of the pipeline at any time that is convenient for you.

Thank you for your work on Hampi. The ruins deserve to be seen whole.

Warm regards,

**Nikita Gupta**
Head of Data Science, super.money

**Yashvardhan Gupta**
Founding Member, super.money

---

*Project repository: github.com/akathedatascienceguy/hampi-revived*

---
> *"The ruins are not the absence of the empire — they are its most durable signature."*
