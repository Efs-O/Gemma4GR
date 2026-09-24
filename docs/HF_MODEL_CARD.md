---
language:
- el
- en
base_model:
- google/gemma-4-E4B-it
tags:
- greek
- qa
- lora
- gguf
- gemma
- gemma4
- unsloth
- llama.cpp
license: gemma
---

# Gemma4GR — Greek fine-tune of Gemma 4 E4B (v3)

> **⚠️ Correction notice (2026-09-24).** The v2 GGUFs uploaded here on 2026-05-15 were exported incorrectly and contained the **unmodified base model** (`google/gemma-4-E4B-it`). I verified this tensor by tensor. **v3 is the first GGUF build with the fine-tune actually merged in.** The LoRA adapters were unaffected; only the export was broken.
>
> - The evaluation numbers previously shown on this card (+29% overall, +67% audio) compared two builds of the base model. **They are withdrawn** and replaced by the measurements below.
> - The old v2 files have been removed from the main branch. They remain in this repository's history (commit `6ac9ec4`) for anyone who wants to verify.
> - Thanks to everyone who tested v2 and reported problems. That is how this was caught.

> **Experimental.** This is a small hobby fine-tune. Read *Limitations* before using it.

**Code:** https://github.com/Efs-O/Gemma4GR

---

## What is this?

Gemma4GR v3 is [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it) with one LoRA adapter merged in, trained on ~2,200 curated Greek question–answer pairs. The goal is cleaner, more natural Modern Greek.

What it changes, measured against stock Gemma 4 E4B under identical settings:

- **Far fewer "strange characters".** Stock E4B puts stray foreign-script fragments (Cyrillic, Korean, Arabic…) into about 40% of Greek answers. v3 does so in about 1%.
- **It stops when it's done.** Stock E4B usually keeps generating until the length limit. v3 gives a complete, focused answer and stops.
- **Answers are closer to reference answers.** The text similarity score (chrF) rises from about 0.30 to 0.39.

What it does **not** change:

- **Knowledge.** v3 doesn't know more than the base model. It still states wrong facts, now in short, confident sentences. See *Limitations*.
- **Audio and vision.** v3 contains no speech or vision fine-tune. The projector (`mmproj`) is the stock one, and speech transcription performs at stock level (measured below).

---

## Files

| File | Size | SHA-256 |
|---|---|---|
| `gemma-4-e4b-it-gr-v3-Q4_K_M.gguf` | 5.34 GB | `a4a4374f9051c481ff91c71095f6083adf1b7f7042d00ac0a3a7a3d0f52dda0d` |
| `gemma-4-e4b-it-gr-v3-Q6_K.gguf` | 6.22 GB | `2950b74b0bbcb48dcd739f7ed38ad5b1ae5aca83638b46849cf57749b0f5de8d` |
| `gemma-4-e4b-it-gr-v3-Q8_0.gguf` | 8.03 GB | `ce9b493a66c4aca21fc5a86ad1d0def895fcb1a3b2bf9dc61e71480c98d0cd32` |
| `gemma-4-e4b-it-gr-v3-mmproj.gguf` | 0.99 GB | `5d22a69ddd1229e475d935fb7d5f7253556f90387191393dee0cb01033c9aab4` |

- **Q4_K_M** is the lightest. **Q6_K** is a good middle ground. **Q8_0** is the closest to full precision.
- The `mmproj` (F16) is only needed for audio or image input. Its tensors are byte-identical to the stock Gemma 4 E4B projector; only the name in its metadata differs.
- The checksums are also in `SHA256SUMS.txt`.

---

## Quick start

### llama.cpp

```bash
llama-server -m gemma-4-e4b-it-gr-v3-Q4_K_M.gguf --mmproj gemma-4-e4b-it-gr-v3-mmproj.gguf --jinja -c 4096 -ngl 99
```

Leave out `--mmproj` for text only. The model was trained and evaluated with this Greek system prompt, and works best with it:

```
Απάντησε μόνο στα Ελληνικά, με φυσική, σωστή και καθαρή γλώσσα. Μην χρησιμοποιείς Αγγλικά, Ρωσικά ή άλλη γλώσσα, εκτός αν το ζητά ρητά η ερώτηση. Δώσε άμεση, ουσιαστική και πλήρη απάντηση.
```

### Ollama

The included `Modelfile` builds a text-only model locally:

```bash
ollama create gemma4gr-v3 -f Modelfile
```

Or pull the v3 build from the Ollama registry (updated 2026-09-24; tags `latest` = `q4_k_m`, `q6_k`, `q8_0`, text only):

```bash
ollama run efso/gemma-4-e4b-it-gr-v2:q6_k
```

If you pulled these tags before 2026-09-24, pull again: the old ones were the base model.

---

## Evaluation

**Setup**
- Frozen held-out sets that never overlap with the training or validation data.
- llama.cpp b11095 `llama-server`, same settings for every model: temperature 0, seed 42, `-c 4096`, max 512 tokens for text and 256 for audio.
- The same Greek system prompt for every model.
- Stock = `google/gemma-4-E4B-it`, converted and quantized with the same pipeline.

| Model | Text chrF ↑ | Answers with foreign-script garbage ↓ | Didn't stop (of 150) ↓ | Garbage-provoking prompts with garbage (of 30) ↓ | Speech CER ↓ | v3 better / stock better (per question) |
|---|---|---|---|---|---|---|
| stock E4B Q4_K_M | 0.298 | 38.7% | 131 | 13 | 0.256 | — |
| stock E4B Q8_0 | 0.307 | 40.0% | 119 | 9 | 0.135 | — |
| **v3 Q4_K_M** | **0.391** | **1.3%** | **0** | **1** | 0.138 | **136 / 14** |
| **v3 Q6_K** | **0.392** | **0.7%** | 1 | **0** | 0.123 | (no stock Q6 run) |
| **v3 Q8_0** | **0.391** | **0.7%** | **0** | **1** | 0.158 | **125 / 25** |

- **Text:** 150 held-out Greek questions with reference answers. chrF measures character-level similarity to the reference.
- **Garbage:** stray fragments in a non-Greek script inside Greek text, or runaway output. Stock E4B's average answer is ~1,250 characters, and most run into the token limit. v3's average is ~270.
- **Garbage-provoking prompts:** 30 prompts designed to trigger foreign-script fragments.
- **Speech:** 40 held-out Greek speech clips, scored by character error rate (CER) against the transcript. The v3 speech results are within the noise of stock (95% CIs overlap), as expected with no speech fine-tune.
- **Per-question comparison:** each text answer is paired with stock's answer to the same question, and the better chrF wins.
- **Factual spot check:** 25 answers, read by hand against the references. v3 had confident factual errors in 8 of 25; stock had clear errors in at least 6 of 25, plus garbage and rambling. v3 fixes the *form* of the answers, not their *knowledge*.
- **Export check:** every GGUF was checked tensor by tensor to confirm it contains the fine-tune, i.e. differs from the base exactly where the adapter changes it.

---

## Limitations

- **Not a source of facts.** v3 writes fluent, confident Greek, and it is often wrong: dishes, myths and historical details get mixed up. Treat it as a Greek *language* model and verify anything factual.
- **Garbage is reduced, not eliminated.** About 1 answer in 100 still contains a stray foreign fragment, e.g. a Hebrew or Korean syllable inside a Greek word. Occasionally it also invents a non-word.
- **Small training set.** ~2,200 Q&A pairs. A bigger quality jump needs more good Greek data.
- **Not ready for unsupervised use by children.** That is the long-term goal of the project (Gemma4Kids), but the factual error rate rules it out for now.

---

## Training details (v3)

| Parameter | Value |
|---|---|
| Base model | `google/gemma-4-E4B-it` |
| Method | QLoRA 4-bit with [Unsloth](https://github.com/unslothai/unsloth) |
| LoRA | r 32, alpha 64, language-model attention + MLP projections |
| Data | 2,219 training / 96 validation Greek Q&A pairs, generated synthetically with large language models, then deduplicated and filtered (non-Greek-script rows and near-duplicates removed) |
| Loss | response-only (the model learns the answers, not the prompts) |
| Schedule | 2 epochs, 1,110 steps, effective batch 4, LR 1e-4, 20 warmup steps, AdamW 8-bit |
| Final validation loss | 0.960 |
| Hardware | 1× NVIDIA RTX 5060 Ti (16 GB) |

### Training data (not released)

The dataset is not public. To make it verifiable, these are the SHA-256 fingerprints of the exact files used (UTF-8 JSONL, LF line endings). Anyone given the files later can check that they are the ones behind this model.

| File | Rows | SHA-256 |
|---|---:|---|
| `train.jsonl` | 2,219 | `88ef474a890bfb4148029cc7510a5d1a824d777b908d8e9329526a2f1b6e0697` |
| `val.jsonl` | 96 | `16ea1210e155a2d216ee00d7ba940f122a2fef338f963ca4420bc380e6ef600e` |
| `text_eval.jsonl` (held-out eval, never trained on) | 150 | `ba5749cf5b96dbb454417e74d34d5f848a51d830ffc58153c1d8a1267ca7f715` |
| `garbage_probe.jsonl` (held-out eval) | 30 | `f628dd60aa1882452226dbabbf707be331d4ff9c012c8a1f25d6a2be0777f5ef` |

**What's in it**
- Single-turn Greek Q&A: one question, one answer of a few sentences.
- All 2,219 training questions are distinct.
- Median question: 11 words. Median answer: 44 words (80% of answers are 34–53 words). About 97,000 answer words in total.
- Ten topic areas: history, culture, geography, the Greek language, science, everyday life, food, children's education, mythology, and Orthodox religion.
- It was generated synthetically with large language models, then filtered: rows with non-Greek script and near-duplicates (Jaccard ≥ 0.85) were removed.

**Examples** (10 of the 2,219 training rows, hand-picked for readability, shown verbatim)

| Question | Answer |
|---|---|
| Πώς σερβίρεται συνήθως η χωριάτικη σαλάτα; | Η χωριάτικη σαλάτα αποτελείται από ντομάτα, αγγούρι, κρεμμύδι, ελιές και μια μεγάλη φέτα φέτας από πάνω. Τα λαχανικά κόβονται σε χοντρά κομμάτια και περιχύνονται με ελαιόλαδο και ρίγανη. Δεν προστίθεται μαρούλι στην αυθεντική συνταγή των Κυκλάδων. |
| Πώς χρησιμοποιούνται τα εποπτικά μέσα για τη διδασκαλία των εποχών; | Οι δάσκαλοι φέρνουν στην τάξη φυσικά αντικείμενα, όπως ξερά φύλλα ή λουλούδια, για να δείξουν τις αλλαγές στη φύση. Τα παιδιά παρατηρούν τις λεπτομέρειες, όπως τα μικρά αγκάθια σε έναν κάκτο ή την υφή του χιονιού. Αυτή η βιωματική εμπειρία κάνει την αφηρημένη έννοια του χρόνου πιο κατανοητή για τα μικρά μυαλά. |
| Ποια λίμνη αποτελεί το μεγαλύτερο φυσικό υδάτινο σώμα στην Ελλάδα και πού βρίσκεται; | Η λίμνη Τριχωνίδα είναι η μεγαλύτερη φυσική λίμνη της χώρας και βρίσκεται στην Αιτωλοακαρνανία. Τα νερά της φιλοξενούν σπάνια είδη χλωρίδας και πανίδας, ενώ οι όχθες της καλύπτονται από πυκνή βλάστηση. Παρόλο που η περιοχή είναι πλούσια σε ζωή, κάποιες απομακρυσμένες γωνίες μοιάζουν σχεδόν εγκαταλειμμένες από τον σύγχρονο τουρισμό. |
| Τι γνωρίζουμε για τη λίμνη Κερκίνη; | Πρόκειται για έναν τεχνητό ταμιευτήρα στα σύνορα με τη Βόρεια Μακεδονία, ο οποίος εξελίχθηκε σε σημαντικό υδροβιότοπο. Φιλοξενεί σπάνια είδη πουλιών και αποτελεί πρότυπο οικολογικής προστασίας για όλη την Ευρώπη. |
| Με ποιο παιχνίδι εξασκούνται τα παιδιά στην αναγνώριση των γραμμάτων του αλφαβήτου; | Ένα δημοφιλές παιχνίδι είναι η αναζήτηση καρτών με γράμματα κρυμμένες στην τάξη ή την αυλή. Τα παιδιά ενθουσιάζονται όταν βρίσκουν το κάθε γράμμα και το τοποθετούν στη σωστή σειρά. Αυτή η δραστηριότητα μετατρέπει τη μάθηση σε μια ευχάριστη περιπέτεια γεμάτη κίνηση. |
| Τι ρόλο παίζουν οι εικόνες με ζώα στη διδασκαλία των πρώτων γραμμάτων; | Οι ζωγραφιές με ζώα βοηθούν τα παιδιά να συνδέσουν τους ήχους με οικείες έννοιες, όπως το «α» για την αλεπού. Αυτή η μέθοδος κάνει την εκμάθηση του αλφαβήτου πιο διασκεδαστική και εύκολη για όλους. Έτσι, τα μικρά παιδιά περιμένουν με ευχαρίστηση το επόμενο μάθημα. |
| Ποιος ήταν ο βασικός σκοπός της Ιωνικής Επανάστασης έναντι των Περσών; | Οι ελληνικές πόλεις της Ιωνίας επιδίωξαν να αποτινάξουν τον περσικό ζυγό και να κερδίσουν την αυτονομία τους. Η εξέγερση αυτή λειτούργησε ως η αφορμή για τους μετέπειτα Περσικούς Πολέμους. Παρόλο που τελικά καταπνίγηκε, η ψυχή των επαναστατών ενέπνευσε τους υπόλοιπους Έλληνες. |
| Ποια είναι η σημασία της απλοποίησης της γραμματικής στη μετάβαση από την αρχαία στη νέα ελληνική; | Η νέα ελληνική απλοποίησε σημαντικά το κλιτικό σύστημα, καταργώντας τον δυϊκό αριθμό και μειώνοντας τις πτώσεις. Αυτή η εξέλιξη διευκόλυνε την επικοινωνία και συνέβαλε στην επικράτηση της δημοτικής ως επίσημης γλώσσας του κράτους. Η διαδικασία αυτή ήταν φυσική και απαραίτητη για τη ζωντάνια της γλώσσας στους αιώνες. |
| Ποιες ήταν οι άμεσες συνέπειες του Εμφυλίου Πολέμου για την κοινωνία και την οικονομία της μεταπολεμικής Ελλάδας; | Ο εμφύλιος άφησε πίσω του μια χώρα κατεστραμμένη, με τεράστιες υλικές ζημιές και βαθιά κοινωνικά τραύματα που δίχασαν τον λαό για δεκαετίες. Χιλιάδες άνθρωποι σκοτώθηκαν ή αναγκάστηκαν να εγκαταλείψουν τις εστίες τους, ενώ η οικονομία βρισκόταν σε πλήρη κατάρρευση. Η περίοδος αυτή καθυστέρησε σημαντικά την ανοικοδόμηση και την ομαλή πολιτική λειτουργία του νεοελληνικού κράτους. |
| Τι διακρίνει το Άγιον Όρος ως μοναδικό πνευματικό κέντρο στον ορθόδοξο κόσμο; | Αυτή η χερσόνησος φιλοξενεί είκοσι ιερές μονές όπου βιώνεται αδιάκοπα η προσευχή εδώ και αιώνες. Η αυστηρή ασκητική ζωή των μοναχών διατηρεί αρχαίες παραδόσεις μακριά από τον κοσμικό θόρυβο. Πολλοί προσκυνητές ταξιδεύουν εκεί για να βρουν γαλήνη και να προσκυνήσουν θαυματουργές εικόνες. |

The data teaches **form**: natural, clean, well-formed Greek answers. It is not a fact source, and some synthetic answers contain factual mistakes. See Limitations.

**Merge and export**
- The adapter is merged on the CPU at tensor level: W + (α/r)·B·A computed in fp32, stored in bf16.
- Converted and quantized with llama.cpp b11095.
- The chat template is Gemma 4's own, unchanged.

**What changed from v2 besides the export fix**
- The QA data no longer includes ~3,400 voice-recording sentences under generic prompts.
- Loss is response-only.
- The data was deduplicated.
- v3 contains no speech adapter.

---

## License

A fine-tune of [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it), governed by the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).

---

## Citation

```
@misc{gemma4gr2026,
  title={Gemma4GR: Fine-tuning Gemma 4 for Greek},
  author={Efstathios Outas},
  year={2026},
  note={v3. Experimental.},
  url={https://huggingface.co/Efso/gemma-4-E4B-it-GR-v2}
}
```

---

## Acknowledgements

- [Unsloth](https://github.com/unslothai/unsloth), for the QLoRA fine-tuning infrastructure.
- [Google DeepMind](https://deepmind.google), for the Gemma 4 base model.
- [llama.cpp](https://github.com/ggml-org/llama.cpp), for the GGUF conversion and inference.
- Chara Kaltsou, creator of the JOY Greek voice (Piper TTS), used in the project's earlier speech experiments.
- The Google Gemma 4 Good Hackathon (Kaggle, May 2026), where this project started.
- Everyone who tested v2 and reported problems.
