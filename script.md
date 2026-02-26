# BDH Interpretability Suite — Demo Video Script

> **Competition:** KRITI 2026 · AI Interpretability Challenge  
> **Project:** BDH Neural Observatory — An Interactive Interpretability Suite for the BDH Post-Transformer Architecture  
> **Estimated Duration:** 8–10 minutes

---

## INTRO (0:00 – 0:40)

**[Screen: Browser tab with the deployed site URL]**

> "Hey everyone! In this video I'll walk you through the BDH Interpretability Suite — our submission for the KRITI 2026 AI Interpretability Challenge."
>
> "BDH stands for a post-transformer architecture that takes a fundamentally different approach to neural networks. Instead of dense, opaque activations, BDH is sparse by design — only about 5 percent of neurons fire at any time — and that architectural choice makes it naturally interpretable."
>
> "We've built a full interactive web app with eight different lenses to explore this architecture. Let me show you each one."

---

## PAGE 1 — HOME / OBSERVATORY (0:40 – 1:30)

**[Screen: Landing page with the hero section, animated background, heatmap]**

> "This is the home page — the Observatory. Up top you see the main hero with the animated wireframe terrain and the BDH branding for KRITI 2026."

**[Scroll to the activation signature heatmap]**

> "Here's a live activation signature — this is real neuron data from our trained model. Each row is a layer, each cell is a sampled neuron, and the green intensity shows activation strength. Notice how sparse it is — most cells are dark. The heatmap auto-cycles through different input tokens every 3 seconds, and you can click on individual token buttons to inspect them."

**[Scroll to the stats bar]**

> "These four stats summarize what makes BDH special: roughly 5% neuron activation versus 95% in standard transformers, linear O(T) attention instead of quadratic, infinite context length through constant memory, and a one-to-one mapping between synapses and concepts."

**[Scroll to the core equations]**

> "These are the four core equations that define each BDH layer — sparse activation, causal attention, gated output, and Hebbian update. Every section of this app ties back to one of these operations."

**[Scroll to the feature grid]**

> "And here's our navigation grid — eight interactive lenses, each exploring a different aspect of the architecture. Let's go through them one by one."

---

## PAGE 2 — STRUCTURAL VIEW / ARCHITECTURE (1:30 – 2:30)

**[Click "Structural View" or navigate via sidebar → Architecture]**

> "The Architecture page gives you an animated, step-by-step walkthrough of how data flows through a BDH layer."

**[Type or keep the default input text, click Play]**

> "I can type any input sentence here. When I hit play, the animation walks through all 13 steps of the pipeline — from raw byte embedding, through the sparse encoder, attention, gating, all the way to the final prediction."

**[Let animation play for a few steps, then pause]**

> "Each step highlights the active component in the architecture diagram on the left. On the right, you get the math detail panel showing the actual equation, dimensions, and what the operation does."

**[Click on a specific block in the diagram]**

> "I can also click any block to jump directly to it. You can step forward and backward token by token, and the animation shows how each token flows sequentially through the layers."

**[Point out the token selector and layer indicator]**

> "The token bar at the top shows which character is being processed, and the layer indicator tracks depth. This is precomputed playback data from real model inference — not a simulation."

---

## PAGE 3 — SPARSITY VIEW / SPARSE BRAIN COMPARATOR (2:30 – 3:30)

**[Navigate to Sparsity via sidebar]**

> "The Sparsity page is where we directly compare BDH's activation patterns against standard transformers."

**[Show the default precomputed view]**

> "On the left, the BDH panel shows the measured sparsity — around 94–95% of neurons are silent. On the right, the Transformer reference shows only about 5–8% sparsity. The contrast is dramatic."

**[Point to the neuron grids]**

> "These grids visualize individual neurons — green dots are active, dark cells are silent. In BDH, you see a vast sea of dark cells with scattered green. In the Transformer panel, it's almost entirely lit up."

**[Switch model to Portuguese or Merged from the dropdown]**

> "I can switch between the French specialist, Portuguese specialist, or the merged model. Each has its own default sentence in the appropriate language."

**[Point to the per-layer breakdown]**

> "Below is the per-layer sparsity breakdown. You can see how sparsity varies across layers — some layers are more sparse than others, but they all maintain high sparsity. The token heatmap lets me hover over individual tokens to see which layers activate most."

**[Note the Live vs Precomputed badge]**

> "When the backend is available, this runs live inference and shows 'Measured' data. Otherwise it gracefully falls back to precomputed results."

---

## PAGE 4 — TOPOLOGY VIEW / 3D GRAPH (3:30 – 4:30)

**[Navigate to Graph via sidebar]**

> "The Topology View renders a 3D force-directed graph of neural connectivity. This uses WebGL and Three.js to visualize how neurons cluster."

**[Let the graph load and settle]**

> "Each node is a neuron, colored by its cluster assignment. The edges represent co-activation strength — neurons that fire together are linked. You can see clear cluster formation, which reflects the model learning structured representations."

**[Rotate the graph by dragging, zoom in/out]**

> "I can rotate, zoom, and pan the 3D view. Clicking on a cluster expands it to show its member neurons."

**[Point to the head selector and beta slider]**

> "There are controls for switching between the 4 attention heads and adjusting the beta threshold — which controls the minimum edge weight for a connection to be shown. Higher beta means only the strongest connections are displayed."

**[Click on a cluster]**

> "When I click a cluster, the sidebar shows statistics about that cluster — its size, the distribution of activation strengths, and which neurons are the hubs."

---

## PAGE 5 — CONCEPT VIEW / MONOSEMANTICITY (4:30 – 6:00)

**[Navigate to Monosemanticity via sidebar]**

> "This is the biggest section of our app — the Monosemanticity page. This is where we prove that individual neurons and synapses in BDH reliably encode specific concepts."

**[Show the concept selector and layer selector at the top]**

> "At the top, I can pick a concept category — like Currency, Number, Punctuation, or Legal terms — and select which layer to analyze. Below that are six analysis tabs."

### Tab 1: Synapse Tracking

**[Click Synapse Tracking tab]**

> "Synapse Tracking shows how individual synapses — the σ(i,j) co-activation values — grow over time as the model reads through text. If a synapse spikes at, say, currency symbols like € and $ but stays flat at other words, that synapse has learned the currency concept."

**[Hover over the synapse chart, point out the spikes at concept words]**

> "You can see the green markers at concept words where the synapse fires strongly, and flat regions everywhere else. This is the Hebbian 'neurons that fire together, wire together' principle in action."

### Tab 2: Selectivity

**[Click Selectivity tab]**

> "Selectivity gives us statistical proof. A selectivity score of 1.0 means a neuron fires exclusively for one concept. We show the distribution histogram — most neurons have low selectivity, but a meaningful fraction are highly selective."

**[Point to the histogram and the per-concept table]**

> "The Mann-Whitney U test confirms these results are statistically significant with p-values below 0.05. The table below lists the top monosemantic neurons per concept with their selectivity scores."

### Tab 3: Sparse Fingerprinting

**[Click Sparse Fingerprinting tab]**

> "Sparse Fingerprinting asks: if the model truly encodes concepts, words from the same category should produce similar sparse activation patterns. We measure cosine similarity between their sparse vectors."

**[Show the similarity matrix heatmap]**

> "This heatmap shows pair-wise similarity between words in the selected category. High similarity within a category — shown by the warm colors on the diagonal — confirms consistent concept encoding."

### Tab 4: Cross-Concept

**[Click Cross-Concept tab]**

> "Cross-Concept analysis checks whether different concepts are cleanly separated. We compare the top active neurons between categories."

**[Point to the distinctness bars]**

> "High distinctness means the model uses different neurons for different ideas. Low overlap is a sign of genuine structure, not noise."

### Tab 5: Shared Neurons

**[Click Shared Neurons tab]**

> "The Shared Neurons view lets me pick a reference word and see exactly which of its top neurons also fire for other same-concept words. Green cells mean shared — providing a visual fingerprint of how concepts are encoded."

### Tab 6: Neuron Graph

**[Click Neuron Graph tab]**

> "Finally, the Neuron Graph renders a force-directed network linking words to their top neurons. Hub neurons — the ones connected to multiple words from the same category — are the 'concept neurons' that BDH has learned."

### Try It Yourself — Category Affinity

**[Scroll down to the interactive probe section]**

> "At the bottom there's a 'Try It Yourself' section. You can type any word and the system computes its category affinity — how similar its activation pattern is to each known concept category. This uses sparse cosine similarity against the precomputed concept fingerprints."

---

## PAGE 6 — DYNAMICS VIEW / HEBBIAN LEARNING (6:00 – 6:50)

**[Navigate to Hebbian via sidebar]**

> "The Hebbian Learning page visualizes how memory forms during a single forward pass."

**[Show the input field with a French sentence]**

> "I type a French sentence and hit Analyze. The backend runs the model and captures σ values — the co-activation signals — at every word position."

**[Show the playback controls, click Play]**

> "Then I can play it back word-by-word. The visualization shows gate activity, synapse deltas — how much each synapse changed at each word — and cumulative sigma values."

**[Point to the layer/head selector]**

> "I can switch between layers and heads to see how different parts of the network respond. Some layers show strong Hebbian learning, others are quieter."

**[Toggle between Δ sigma and cumulative σ]**

> "Toggling between delta and cumulative view shows the instantaneous change versus the accumulated memory. This is the 'neurons that fire together, wire together' principle visualized in real time."

---

## PAGE 7 — COMPOSITION VIEW / MODEL MERGING (6:50 – 8:00)

**[Navigate to Merge via sidebar]**

> "The Model Merging page demonstrates one of BDH's most powerful properties — compositional intelligence. We trained two specialist models separately — one on French, one on Portuguese — and merged them by concatenating their neuron spaces."

**[Point to the animated merge diagram]**

> "This animated diagram shows the merge process step by step: two specialists come in, their neuron spaces are concatenated, and an optional fine-tuning step recovers quality."

**[Scroll to Training Evolution charts]**

> "The Training Evolution section shows loss and sparsity curves during training — you can toggle between the two."

**[Scroll to Model Cards]**

> "These model cards show the architecture details for each specialist and the merged model — parameter counts, vocabulary, neuron dimensions."

**[Scroll to Loss Comparison table]**

> "The loss comparison table shows how each model performs on both languages. The specialists excel at their own language, the merged model handles both."

**[Scroll to Sample Generations]**

> "Sample Generations show the same prompt fed through all model variants — you can switch between different prompts and compare the outputs side by side."

**[Scroll to Neuron Heritage Map]**

> "The Heritage Map visualizes which neurons in the merged model trace back to which specialist — the first half comes from French, the second half from Portuguese."

**[Scroll to Precomputed Heritage Probe]**

> "The Heritage Probe section shows how well the merged model routes inputs to their parent-language neurons. When you feed French text, it should primarily activate French-origin neurons, and vice versa for Portuguese."

**[Scroll to Live Generation]**

> "Finally, Live Generation lets you type a prompt and generate text through the merged model in real-time when the backend is available."

---

## PAGE 8 — FINDINGS (8:00 – 8:40)

**[Navigate to Findings via sidebar]**

> "The Findings page brings everything together into a summary dashboard."

**[Point to the hero stats at the top]**

> "The top row shows key metrics at a glance — total neurons analyzed, selective neurons, mean selectivity, and number of concept categories."

**[Scroll through the sections]**

> "Below that we have dedicated sections for the Loss Landscape with bar charts comparing all model variants, Neuron Selectivity with a histogram and radial gauge, Heritage Routing showing how well the merge preserves language-specific pathways, Synapse Tracking with concept-specific sigma growth, Concept Distinctness between categories, and a Neuron Activation Heatmap showing firing patterns across layers."

> "Every chart on this page is driven by real data from our trained checkpoints — nothing is mocked."

---

## PAGE 9 — LEARN BDH / TUTORIAL (8:40 – 9:15)

**[Navigate to Learn BDH via sidebar]**

> "The Learn BDH page is an interactive educational walkthrough of the architecture in 8 steps."

**[Show the sidebar with the 8 steps listed]**

> "Each step covers one building block — starting from Byte Embedding, through Sparse Encoding, RoPE positional encoding, Linear Attention, Value Encoding, Sparse Gating, Decode and Residual, all the way to the Full Layer view."

**[Click through a couple of steps]**

> "For each step you get a description tab explaining the concept in plain language, a theory tab with deeper mathematical details, a key insight callout, and the actual Python code from our implementation."

**[Point to the interactive visualization panel]**

> "On the right side, each step has a custom animated visualization — like seeing neurons light up during sparse encoding, or watching the gating mechanism filter activations. These are generated procedurally from the actual model parameters."

**[Point to the difficulty badges]**

> "Steps are tagged by difficulty — Easy, Medium, or Hard — so you can gauge the complexity before diving in."

---

## TECHNICAL DETAILS & WRAP-UP (9:15 – 9:50)

**[Can show the sidebar with backend status indicator, or briefly mention the tech stack]**

> "A few technical details: the frontend is built with React, TypeScript, and Vite, styled with Tailwind CSS, and uses Framer Motion for all the animations. The 3D graph uses Three.js with react-force-graph. The backend is FastAPI serving the BDH model with PyTorch."
>
> "The architecture itself processes raw UTF-8 bytes — no tokenizer needed — with 6 layers, 4 attention heads, 192 embedding dimensions, and 3072 neurons per head. The entire model is about 27 million parameters."
>
> "Everything you've seen is driven by real data from trained checkpoints — French specialist, Portuguese specialist, and the merged polyglot model."

**[Quick scroll through the sidebar showing all 9 navigation items]**

> "So to recap — we've got the Observatory home page, Structural View for architecture animation, Sparsity View for activation comparison, Topology View for 3D neural graphs, Concept View for monosemanticity proof, Dynamics View for Hebbian learning, Composition View for model merging, a Findings dashboard, and an interactive tutorial."
>
> "Thanks for watching! This is the BDH Interpretability Suite for KRITI 2026."

---

## RECORDING TIPS

- **Resolution:** Record at 1920×1080 for clarity
- **Browser:** Use Chrome/Edge in full-screen (F11) for a clean look
- **Sidebar:** Keep the sidebar visible — it shows where you are in the app
- **Transitions:** Use the sidebar navigation to move between pages smoothly
- **Hover effects:** Move the mouse slowly over interactive elements to show tooltips and hover states
- **Loading states:** The app loads precomputed data first, so pages appear instantly — no awkward loading screens
- **Backend indicator:** The bottom of the sidebar shows if the backend is connected (green) or offline (gray). For the demo, precomputed data works perfectly without a backend
- **Pace:** Pause for 2–3 seconds on each major visualization to let the viewer absorb it
