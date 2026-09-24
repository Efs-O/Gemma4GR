# JOY Voice — Piper Test Recording Guide (20 Sentences)

**Purpose:** Record these 20 sentences to create a minimal dataset for the local Piper smoke test.  
**Goal after recording:** Verify Docker + Piper training pipeline works end-to-end, and measure steps/second to project full training time before committing to Vast.ai.

---

## Recording Setup

| Setting | Requirement |
|---------|-------------|
| Microphone | USB condenser or any quiet-room mic |
| Room | Quiet — no fan, TV, traffic noise |
| Format | WAV, 22050 Hz, mono, 16-bit — **handled automatically by the recorder** |
| Clip length | Read naturally — no rushing, no dramatic pause |
| Filenames | Saved automatically as `joy_test_0001.wav` … `joy_test_0020.wav` |
| Save location | `data/piper_test_20/wavs/` — set automatically via `--out-dir` |

**Use the custom recorder app** — it handles 22050 Hz, speech detection, auto-advance, level meter, and normalization:

```bash
python training/record_human_voice_dataset.py \
  --manifest data/piper_test_20/recording_manifest.csv \
  --out-dir data/piper_test_20
```

The recorder will:
- Show each sentence in large text on screen
- Arm and wait for your voice (speech threshold detection)
- Save the clip automatically when you stop speaking
- Advance to the next sentence automatically
- Show a live input level bar so you can check mic gain

---

## Quick Tips for Quality

- Sit close to the mic (20-30 cm), speak at normal conversation volume
- Say each sentence **once** clearly — the recorder saves on trailing silence
- If you make a mistake, click **Retry Sentence** — it deletes the current take so you can re-record
- Breathe naturally — do not hold breath before speaking
- Keep consistent distance and volume across all 20 clips
- Read the Greek exactly as written — stress marks (τόνος) show the correct emphasis
- Watch the live level bar: if it barely moves, increase the **Mic sensitivity** slider; if it constantly maxes out, lower it

---

## The 20 Sentences

Read each sentence naturally, as if speaking to a child or in normal conversation. The sentences cover different emotional tones on purpose — warm, instructive, storytelling, questioning.

| File | Sentence |
|------|---------|
| `joy_test_0001.wav` | Καλημέρα! Είσαι έτοιμος να μάθουμε κάτι καινούριο σήμερα; |
| `joy_test_0002.wav` | Η γάτα κοιμάται στον καναπέ και ονειρεύεται ψάρια. |
| `joy_test_0003.wav` | Βάλε τα παπούτσια σου και πάμε μια βόλτα στο πάρκο. |
| `joy_test_0004.wav` | Μπράβο σου! Έκανες πολύ καλή δουλειά σήμερα. |
| `joy_test_0005.wav` | Στο παραμύθι, η πριγκίπισσα ζούσε σε ένα κάστρο δίπλα στη θάλασσα. |
| `joy_test_0006.wav` | Δύο συν τρία κάνει πέντε· εσύ το ξέρεις αυτό; |
| `joy_test_0007.wav` | Άνοιξε το βιβλίο στη σελίδα δώδεκα και διάβασε μόνος σου. |
| `joy_test_0008.wav` | Θέλεις να φας λίγο φρούτο; Έχουμε μπανάνες και μήλα. |
| `joy_test_0009.wav` | Σήμερα είναι Τετάρτη και έχει ωραίο καιρό έξω. |
| `joy_test_0010.wav` | Προσοχή! Μην τρέχεις στον διάδρομο, μπορεί να πέσεις. |
| `joy_test_0011.wav` | Ο ήλιος δύει στη θάλασσα και ο ουρανός γίνεται πορτοκαλί. |
| `joy_test_0012.wav` | Πες μου, τι έμαθες σήμερα στο σχολείο; |
| `joy_test_0013.wav` | Είναι ώρα για ύπνο· πάμε να σε σκεπάσω και να σου πω ένα παραμύθι. |
| `joy_test_0014.wav` | Στο ζωολογικό κήπο είδαμε ελέφαντες, καμηλοπαρδάλεις και πολύχρωμα παπαγαλάκια. |
| `joy_test_0015.wav` | Το πρωινό είναι έτοιμο· έχω φτιάξει γάλα με δημητριακά και φρέσκο χυμό πορτοκαλιού. |
| `joy_test_0016.wav` | Μπορείς να με βοηθήσεις να βρω το μολύβι μου; Νομίζω ότι το άφησα στο τραπέζι. |
| `joy_test_0017.wav` | Χθες πήγαμε στην παραλία και χτίσαμε το πιο μεγάλο κάστρο από άμμο. |
| `joy_test_0018.wav` | Αν θέλεις να γίνεις καλός στα μαθηματικά, πρέπει να εξασκείσαι κάθε μέρα. |
| `joy_test_0019.wav` | Η γιαγιά ετοίμασε κουραμπιέδες και μελομακάρονα για τα Χριστούγεννα. |
| `joy_test_0020.wav` | Καληνύχτα, γλυκό μου· ονειρεύσου όμορφα και τα λέμε το πρωί. |

---

## After Recording

1. The recorder saves automatically — when done, click **Open Output Folder** to verify
2. Confirm all 20 files are in `data/piper_test_20/wavs/` with names `joy_test_0001.wav` … `joy_test_0020.wav`
3. Run the smoke test:

```bash
# Add to .env temporarily:
# PIPER_DATASET_DIR=data/piper_test_20
# PIPER_MAX_EPOCHS=2

python training/train_piper.py
```

4. Note the wall-clock time. Use this formula to project full run:
   ```
   projected_hours = (3000 / 20) × (20 / 2) × measured_hours
   ```
   - If projected > 12h → use Vast.ai for the full run
   - If projected < 8h → local is viable

5. After smoke test passes, restore `.env` to `PIPER_DATASET_DIR=data/human_voice_dataset` and `PIPER_MAX_EPOCHS=20` for the full run.

---

## Why These Sentences

These 20 sentences were chosen to cover:
- **Phonetic diversity:** Different vowel clusters, consonant groups, Greek diphthongs
- **Prosody variety:** Questions, exclamations, statements, compound sentences
- **Register variety:** Warm/children, storytelling, instructive, everyday
- **Practical coverage:** Greeting, numbers spoken aloud, bedtime, school, nature, food

Even with only 20 samples and 2 epochs, the training run will validate the full pipeline. Quality of the output audio from this test will be poor (expected) — the goal is pipeline health, not voice quality.
