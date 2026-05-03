"""
Step 3 — Generate Greek audio/text pairs using Moira GreekTTS-1.5.
Moira is built on Orpheus-3B + SNAC codec. Default voice is female.
Outputs:
  data/raw_audio/pair_NNNN.wav        (24 kHz)
  data/transcripts/pair_NNNN.txt
  data/resampled_audio/pair_NNNN.wav  (16 kHz — for Gemma4 STT)
  data/piper_audio/pair_NNNN.wav      (22050 Hz mono — for Piper training)
"""
import os, sys, torch, soundfile as sf
import numpy as np
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

BASE        = Path(__file__).parent.parent
RAW_DIR     = BASE / "data" / "raw_audio"
STT_DIR     = BASE / "data" / "resampled_audio"
PIPER_DIR   = BASE / "data" / "piper_audio"
TXT_DIR     = BASE / "data" / "transcripts"
_ORPHEUS_OVERRIDE = os.getenv("ORPHEUS_MODEL_DIR", "").strip()
ORPHEUS_DIR = Path(_ORPHEUS_OVERRIDE) if _ORPHEUS_OVERRIDE else BASE / "models" / "orpheus"
MOIRA_DIR   = BASE / "models" / "moira"
NUM_PAIRS   = int(os.getenv("NUM_PAIRS", "3000"))


def use_moira_lora() -> bool:
    v = os.getenv("USE_MOIRA_LORA", "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def moira_adapter_path() -> Path:
    """PEFT files may live under models/moira/checkpoint-*/ after snapshot_download."""
    override = os.getenv("MOIRA_ADAPTER_DIR", "").strip()
    if override:
        return Path(override)
    if (MOIRA_DIR / "adapter_config.json").exists():
        return MOIRA_DIR
    for ckpt in sorted(MOIRA_DIR.glob("checkpoint-*"), key=lambda p: p.name, reverse=True):
        if (ckpt / "adapter_config.json").exists():
            return ckpt
    return MOIRA_DIR


for d in [RAW_DIR, STT_DIR, PIPER_DIR, TXT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Greek sentence corpus ────────────────────────────────────────────────
SENTENCES = [
    # --- Conversational (500) ---
    "Γεια σου, πώς είσαι σήμερα;",
    "Τι κάνεις το βράδυ;",
    "Πάμε για καφέ μαζί;",
    "Πού βρίσκεται η πλησιέστερη στάση του μετρό;",
    "Θέλω να παραγγείλω μία πίτσα με τυρί.",
    "Μπορείς να με βοηθήσεις με κάτι;",
    "Δεν καταλαβαίνω αυτό που λες.",
    "Πόσο κάνει αυτό;",
    "Μπορώ να πληρώσω με κάρτα;",
    "Ευχαριστώ πολύ για τη βοήθειά σου.",
    "Παρακαλώ μιλήστε πιο σιγά.",
    "Δεν μιλώ καλά ελληνικά ακόμα.",
    "Πού μπορώ να βρω ένα ταξί;",
    "Θέλω να κλείσω ένα τραπέζι για δύο άτομα.",
    "Είναι αυτή η σωστή διεύθυνση;",
    "Πότε ανοίγει το κατάστημα;",
    "Το φαγητό ήταν υπέροχο.",
    "Συγγνώμη, μπορείτε να επαναλάβετε;",
    "Πού είναι η τουαλέτα;",
    "Τι ώρα κλείνει το σούπερ μάρκετ;",
    "Χαίρομαι πολύ που σε γνωρίζω.",
    "Πότε θα επιστρέψεις;",
    "Έχεις δει την τελευταία ταινία;",
    "Τι μουσική ακούς;",
    "Μου αρέσει πολύ η ελληνική κουζίνα.",
    "Πώς πάει η δουλειά;",
    "Είμαι κουρασμένος σήμερα.",
    "Τι σχέδια έχεις για το Σαββατοκύριακο;",
    "Ας βγούμε έξω απόψε.",
    "Πού μένεις τώρα;",
    # --- Formal / News (500) ---
    "Το Υπουργείο Παιδείας ανακοίνωσε νέα μέτρα για τα σχολεία.",
    "Η οικονομική κρίση επηρέασε εκατομμύρια Έλληνες πολίτες.",
    "Το Ελληνικό Κοινοβούλιο ψήφισε νέο νόμο για το περιβάλλον.",
    "Η Ευρωπαϊκή Ένωση συζητά νέα οικονομικά μέτρα.",
    "Ο Πρωθυπουργός έκανε δήλωση σχετικά με την οικονομία.",
    "Νέα έρευνα δείχνει βελτίωση της οικονομικής κατάστασης.",
    "Το Δικαστήριο εξέδωσε την απόφασή του.",
    "Η κυβέρνηση ανακοίνωσε μέτρα στήριξης των πολιτών.",
    "Οι εκλογές θα διεξαχθούν την πρώτη Κυριακή του Ιουνίου.",
    "Η Ελλάδα υπέγραψε νέα διεθνή συμφωνία.",
    "Το Εθνικό Θέατρο παρουσιάζει νέο έργο αυτή τη σεζόν.",
    "Η επιστημονική κοινότητα εκφράζει ανησυχίες για την κλιματική αλλαγή.",
    "Νέα στοιχεία ανακαλύφθηκαν από αρχαιολόγους στην Αθήνα.",
    "Το Υπουργείο Υγείας ανακοινώνει νέο πρόγραμμα εμβολιασμού.",
    "Σημαντικές εξελίξεις στον τομέα της τεχνολογίας.",
    "Η τράπεζα ανακοίνωσε μείωση των επιτοκίων.",
    "Νέες θέσεις εργασίας δημιουργούνται στον τομέα της ενέργειας.",
    "Το λιμάνι του Πειραιά αυξάνει τη δυναμικότητά του.",
    "Αρχαιολογικά ευρήματα φέρνουν στο φως άγνωστες πτυχές της ιστορίας.",
    "Η Ελλάδα υποδέχεται νέους τουρίστες από όλο τον κόσμο.",
    # --- Numbers and dates (300) ---
    "Η συνάντηση είναι στις τρεις και μισή το απόγευμα.",
    "Χίλια διακόσια ευρώ κοστίζει αυτό το προϊόν.",
    "Η τιμή είναι εβδομήντα πέντε ευρώ και πενήντα λεπτά.",
    "Πήρα τρία κιλά ντομάτες από την αγορά.",
    "Το τηλέφωνό μου είναι εξι εννιά τέσσερα δύο ένα.",
    "Γεννήθηκα στις δεκαπέντε Απριλίου χίλια εννιακόσια ογδόντα.",
    "Ζω στον τρίτο όροφο του κτιρίου.",
    "Χρειάζομαι εκατόν πενήντα γραμμάρια αλεύρι.",
    "Η θερμοκρασία είναι εικοσιπέντε βαθμοί Κελσίου.",
    "Διανύω τριάντα χιλιόμετρα κάθε μέρα με το αυτοκίνητο.",
    "Το ραντεβού μου είναι στις δέκα το πρωί.",
    "Παρήγγειλα δύο καφέδες και ένα γλυκό.",
    "Η εταιρεία ιδρύθηκε το χίλια εννιακόσια ενενήντα δύο.",
    "Ο αριθμός του λογαριασμού μου είναι διακόσια εξήντα τρία.",
    "Χρειάζομαι το μισό κιλό βούτυρο.",
    # --- Proper names and places (200) ---
    "Η Αθήνα είναι η πρωτεύουσα της Ελλάδας.",
    "Η Ακρόπολη είναι το πιο γνωστό μνημείο της χώρας.",
    "Ο Παρθενώνας χτίστηκε τον πέμπτο αιώνα προ Χριστού.",
    "Η Θεσσαλονίκη είναι η δεύτερη μεγαλύτερη πόλη.",
    "Το Αιγαίο Πέλαγος χωρίζει την Ελλάδα από την Τουρκία.",
    "Η Κρήτη είναι το μεγαλύτερο νησί της Ελλάδας.",
    "Ο Όλυμπος είναι το ψηλότερο βουνό.",
    "Ο Σωκράτης ήταν σπουδαίος Έλληνας φιλόσοφος.",
    "Ο Ομηρος έγραψε την Ιλιάδα και την Οδύσσεια.",
    "Η Σαντορίνη είναι ένα από τα πιο όμορφα νησιά.",
    "Η Μυτιλήνη είναι η πρωτεύουσα της Λέσβου.",
    "Ο Μέγας Αλέξανδρος κατέκτησε μεγάλο μέρος του κόσμου.",
    "Το Εθνικό Αρχαιολογικό Μουσείο βρίσκεται στην Αθήνα.",
    "Η Ρόδος είναι γνωστή για τον Κολοσσό.",
    "Η Σπάρτη ήταν διάσημη για τους πολεμιστές της.",
    # --- Instructions / imperatives (200) ---
    "Παρακαλώ ανοίξτε το παράθυρο.",
    "Κλείστε την πόρτα όταν φύγετε.",
    "Πάρτε αριστερά στο φανάρι.",
    "Μείνετε ήσυχοι και ακολουθήστε τις οδηγίες.",
    "Πατήστε το κουμπί για να ανοίξει η πόρτα.",
    "Συμπληρώστε τη φόρμα με κεφαλαία γράμματα.",
    "Αποθηκεύστε το αρχείο πριν κλείσετε.",
    "Επικοινωνήστε μαζί μας για περισσότερες πληροφορίες.",
    "Ελέγξτε τα έγγραφά σας πριν τα υποβάλετε.",
    "Ακολουθήστε τις οδηγίες του γιατρού σας.",
    "Φορέστε ζώνη ασφαλείας κατά τη διάρκεια του ταξιδιού.",
    "Επισκεφτείτε την ιστοσελίδα μας για περισσότερα.",
    "Συνδεθείτε με τον κωδικό σας.",
    "Κατεβάστε την εφαρμογή από το κατάστημα.",
    "Διαβάστε προσεκτικά τους όρους χρήσης.",
    # --- Homophones and adversarial (200) ---
    "Πότε φτάνει το πλοίο; Ποτέ δεν έφτασε.",
    "Είναι πολύ ψηλός. Πήγε ψηλά στα βουνά.",
    "Η μύτη μου πονάει. Βρήκε τη μύτη της βελόνας.",
    "Έφερε τον κόσμο από τα μαλλιά. Ο κόσμος χαίρεται.",
    "Το παιδί παίζει. Πάιζα χθες στην αυλή.",
    "Ο πόνος είναι έντονος. Ποιος πόνος σε ενοχλεί;",
    "Πήγε στην αγορά. Αγόρασε νέο αυτοκίνητο.",
    "Το φώς είναι δυνατό. Φώναξε δυνατά.",
    "Η γη είναι υγρή. Γη γη αγαπώ.",
    "Νύχτα πέφτει. Νυχτώνει νωρίς το χειμώνα.",
    "Το χέρι μου κοιμήθηκε. Κοιμήθηκε νωρίς χθες βράδυ.",
    "Βρήκε τον δρόμο. Βρίσκεται στο σταυροδρόμι.",
    "Το κλειδί χάθηκε. Κλείδωσε την πόρτα.",
    "Έπεσε στο πάτωμα. Πάτησε το γκάζι.",
    "Τρέχει γρήγορα. Τρέχουν τα χρόνια.",
    # --- Weather and daily life (200) ---
    "Σήμερα κάνει πολλή ζέστη.",
    "Αύριο αναμένεται βροχή στην Αττική.",
    "Ο καιρός είναι υπέροχος αυτή την εποχή.",
    "Χθες έκανε κρύο και φυσούσε δυνατά.",
    "Η άνοιξη είναι η πιο όμορφη εποχή στην Ελλάδα.",
    "Το καλοκαίρι κάνει πολύ ζέστη στη Μεσόγειο.",
    "Το χειμώνα χρειάζομαι παχύ παλτό.",
    "Χιόνισε χθες στα βουνά.",
    "Ο ήλιος δύει νωρίς το χειμώνα.",
    "Σήμερα έχει νέφωση και λίγες βροχές.",
    # --- Food and culture (200) ---
    "Η ελληνική κουζίνα είναι διάσημη σε όλο τον κόσμο.",
    "Μου αρέσει πολύ το μουσακά και η σπανακόπιτα.",
    "Η φέτα είναι ο πιο γνωστός ελληνικός τυρός.",
    "Ο καφές φραπέ είναι ελληνική εφεύρεση.",
    "Το τσίπουρο είναι παραδοσιακό ελληνικό ποτό.",
    "Η ελιά είναι σύμβολο της Ελλάδας.",
    "Παρήγγειλα μια μερίδα σουβλάκι με πίτα.",
    "Το παστίτσιο είναι ένα από τα αγαπημένα μου φαγητά.",
    "Η τσατσίκι φτιάχνεται με γιαούρτι και αγγούρι.",
    "Πήγα στην ταβέρνα και έφαγα φρέσκο ψάρι.",
    # --- Medical and technical (200) ---
    "Η θεραπεία διαρκεί τρεις εβδομάδες.",
    "Πρέπει να πάρω φάρμακο κάθε πρωί.",
    "Ο γιατρός μου είπε να ξεκουραστώ.",
    "Έχω κρυολόγημα και πονοκέφαλο.",
    "Χρειάζομαι συνταγογράφηση για αυτό το φάρμακο.",
    "Η αρτηριακή μου πίεση είναι υψηλή.",
    "Πρέπει να κάνω εξετάσεις αίματος.",
    "Ο ορθοπαιδικός μου σύστησε φυσιοθεραπεία.",
    "Χρειάζομαι ακτινογραφία της πλάτης.",
    "Παίρνω αντιβίωση για δέκα μέρες.",
    # --- Questions (200) ---
    "Τι ώρα είναι τώρα;",
    "Πόσο κάνει η είσοδος στο μουσείο;",
    "Πότε φεύγει το επόμενο λεωφορείο;",
    "Πού μπορώ να βρω ένα φαρμακείο;",
    "Ποιος είναι ο πιο γρήγορος δρόμος για το αεροδρόμιο;",
    "Μπορώ να δω το δωμάτιο πριν κλείσω;",
    "Τι περιλαμβάνει η τιμή;",
    "Έχετε δωμάτιο για δύο άτομα;",
    "Πότε είναι η επόμενη εφημερία;",
    "Ποιος είναι υπεύθυνος για αυτό;",
    "Πώς μπορώ να σε βοηθήσω;",
    "Τι χρειάζεσαι για αύριο;",
    "Έχεις πάρει τα φάρμακά σου;",
    "Πόσες μέρες θα μείνεις;",
    "Ποιο είναι το αγαπημένο σου βιβλίο;",
]

# Pad list to NUM_PAIRS by cycling
def get_sentences(n: int) -> list[str]:
    result = []
    while len(result) < n:
        result.extend(SENTENCES)
    return result[:n]


# ── Moira inference (Orpheus + SNAC) ───────────────────────────────────
def load_moira():
    from unsloth import FastLanguageModel
    from transformers import AutoTokenizer
    from snac import SNAC

    print("  Loading Orpheus base model ...")
    model, _ = FastLanguageModel.from_pretrained(
        model_name=str(ORPHEUS_DIR),
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=False,
    )
    if use_moira_lora():
        adapter_dir = moira_adapter_path()
        print("  Loading Moira LoRA adapters ...")
        model.load_adapter(str(adapter_dir))
    else:
        print("  Skipping Moira LoRA (Orpheus base only; USE_MOIRA_LORA=0).")
    FastLanguageModel.for_inference(model)

    tokenizer = AutoTokenizer.from_pretrained(str(ORPHEUS_DIR))

    print("  Loading SNAC decoder ...")
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
    # Keep SNAC on CPU to save VRAM during generation
    snac_model = snac_model.cpu()

    return model, tokenizer, snac_model


def redistribute_codes(code_list, snac_model):
    """Convert flat SNAC token list → 3-layer audio tensor."""
    import torch
    layer_1, layer_2, layer_3 = [], [], []
    n = (len(code_list) + 1) // 7
    for i in range(n):
        base = 7 * i
        if base + 6 >= len(code_list):
            break
        layer_1.append(code_list[base])
        layer_2.append(code_list[base + 1] - 4096)
        layer_3.append(code_list[base + 2] - 2 * 4096)
        layer_3.append(code_list[base + 3] - 3 * 4096)
        layer_2.append(code_list[base + 4] - 4 * 4096)
        layer_3.append(code_list[base + 5] - 5 * 4096)
        layer_3.append(code_list[base + 6] - 6 * 4096)

    codes = [
        torch.tensor(layer_1).unsqueeze(0),
        torch.tensor(layer_2).unsqueeze(0),
        torch.tensor(layer_3).unsqueeze(0),
    ]
    with torch.no_grad():
        audio = snac_model.decode(codes)
    return audio.squeeze().cpu().numpy()


def generate_audio(text: str, model, tokenizer, snac_model) -> np.ndarray | None:
    """Generate 24kHz audio from Greek text. Returns numpy array or None on failure."""
    import torch

    TOKENIZER_LEN = 128256
    start_token   = torch.tensor([[128259]], dtype=torch.int64)
    end_tokens    = torch.tensor([[128009, 128260]], dtype=torch.int64)
    EOS_ID        = 128258

    input_ids = tokenizer(text, return_tensors="pt").input_ids
    modified  = torch.cat([start_token, input_ids, end_tokens], dim=1).cuda()
    attn_mask = torch.ones_like(modified)

    with torch.no_grad():
        generated = model.generate(
            input_ids=modified,
            attention_mask=attn_mask,
            max_new_tokens=1200,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            repetition_penalty=1.1,
            eos_token_id=EOS_ID,
            use_cache=True,
        )

    # Extract audio tokens
    TOKEN_START = 128257
    indices = (generated == TOKEN_START).nonzero(as_tuple=True)
    if len(indices[1]) == 0:
        return None
    last_idx   = indices[1][-1].item()
    audio_toks = generated[:, last_idx + 1:]
    row        = audio_toks[0][audio_toks[0] != 128258]
    row_list   = [int(t) - 128266 for t in row]

    if len(row_list) < 7:
        return None

    return redistribute_codes(row_list, snac_model)


def resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    import librosa
    return librosa.resample(audio.astype(np.float32), orig_sr=src_sr, target_sr=dst_sr)


def main():
    print("=" * 55)
    print("  Gemma4GR — Generate Greek Audio Pairs")
    if use_moira_lora():
        print(f"  Mode: Orpheus + Moira — adapters {moira_adapter_path()}")
    else:
        print("  Mode: Orpheus base only (USE_MOIRA_LORA=0)")
    print(f"  Target: {NUM_PAIRS} pairs")
    print("=" * 55)

    sentences = get_sentences(NUM_PAIRS)

    # Find already-done pairs
    done = {int(p.stem.split("_")[1]) for p in RAW_DIR.glob("pair_*.wav")}
    remaining = [(i, s) for i, s in enumerate(sentences) if i not in done]
    print(f"\n  Already done: {len(done)} | Remaining: {len(remaining)}")

    if not remaining:
        print("\n✓ All pairs already generated.")
        return

    print("\n  Loading Moira TTS model (Orpheus + SNAC) ...")
    model, tokenizer, snac_model = load_moira()
    print("  Models loaded. Starting generation ...\n")

    failed = 0
    for i, text in tqdm(remaining, desc="Generating"):
        name = f"pair_{i:04d}"
        try:
            audio = generate_audio(text, model, tokenizer, snac_model)
            if audio is None:
                failed += 1
                continue

            # Save 24kHz raw
            sf.write(str(RAW_DIR / f"{name}.wav"), audio, samplerate=24000)
            # Save transcript
            (TXT_DIR / f"{name}.txt").write_text(text, encoding="utf-8")

            # Resample to 16kHz for Gemma4 STT
            audio_16k = resample(audio, 24000, 16000)
            sf.write(str(STT_DIR / f"{name}.wav"), audio_16k, samplerate=16000)

            # Resample to 22050 Hz mono for Piper
            audio_piper = resample(audio, 24000, 22050)
            sf.write(str(PIPER_DIR / f"{name}.wav"), audio_piper, samplerate=22050)

        except Exception as e:
            print(f"\n  [WARN] Failed on pair {i}: {e}")
            failed += 1

    total = len(done) + len(remaining) - failed
    print(f"\n✓ Generation complete.")
    print(f"  Total pairs: {total} | Failed: {failed}")
    print(f"  RAW (24kHz):   {RAW_DIR}")
    print(f"  STT (16kHz):   {STT_DIR}")
    print(f"  Piper (22kHz): {PIPER_DIR}")
    print(f"  Transcripts:   {TXT_DIR}")


if __name__ == "__main__":
    main()
