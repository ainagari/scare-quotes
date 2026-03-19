"""
reconstruct_from_standoff.py
============================
Reconstructs the full annotation JSON (with Reddit dialogue text) from:
  1. standoff_annotations.json   — standoff release (no Reddit text)
  2. quotations_winningargs_info_public.tsv — positional info (no utterances)
  3. The WinningArguments corpus, downloaded via convokit

The reconstructed dialogue HTML is byte-for-byte identical to the original
used during annotation.

Requirements
------------
    pip install convokit pandas

Usage
-----
    python reconstruct_from_standoff.py

The script will download the WinningArguments corpus via convokit on first
run (~few hundred MB).  Subsequent runs use the cached copy.

Output
------
    reconstructed_annotations.json  — full annotation JSON ready for use
                                       in the statistics notebook
"""

import json
import re
from pathlib import Path

import pandas as pd
from convokit import Corpus, download

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent

INPUT_STANDOFF = BASE_DIR / "standoff_annotations.json"
INPUT_TSV      = BASE_DIR / "quotations_winningargs_info_public.tsv"
OUTPUT_JSON    = "../annotated_corpus.json"


# ---------------------------------------------------------------------------
# Helper functions (identical logic to the original extraction script)
# ---------------------------------------------------------------------------

def get_conv_id(item_id: str) -> str:
    """Return the conversation ID (first two '_'-separated components)."""
    return "_".join(item_id.split("_")[:2])


def modify_for_citation(utterance_text: str) -> str:
    """Wrap Reddit citation blocks ('>…\\n\\n') with [STA-CITE]/[END-CITE] tags."""
    citation_matches = list(re.finditer(r"&gt;.*?\n\n", utterance_text))
    if not citation_matches:
        return utterance_text

    filtered_text = ""
    decalage = 0
    next_first_index = 0

    for m in citation_matches:
        start, end = m.span()
        filtered_text += (
            utterance_text[next_first_index:start]
            + "[STA-CITE]"
            + utterance_text[start:end]
            + "[END-CITE]"
        )
        decalage += len("[STA-CITE]") + len("[END-CITE]")
        next_first_index = len(filtered_text) - decalage

    filtered_text += utterance_text[next_first_index:]
    return filtered_text


def strip_citation_markers(text: str, start: int = None, end: int = None):
    """Remove [STA-CITE] and [END-CITE] markers from text, adjusting offsets.

    These markers were added by modify_for_citation() to bracket Reddit
    citation lines (&gt; …) during annotation.  They are not part of the
    original Reddit text and should be removed for plain-text use.

    Parameters
    ----------
    text  : string that may contain [STA-CITE] / [END-CITE] markers
    start : optional character offset to adjust (e.g. quote start position)
    end   : optional character offset to adjust (e.g. quote end position)

    Returns
    -------
    cleaned_text                          if start/end are not provided
    (cleaned_text, adjusted_start, adjusted_end)  otherwise
    """
    markers = ["[STA-CITE]", "[END-CITE]"]
    if start is None and end is None:
        cleaned = text
        for marker in markers:
            cleaned = cleaned.replace(marker, "")
        return cleaned

    shift_start = 0
    shift_end = 0
    for marker in markers:
        for match in re.finditer(re.escape(marker), text):
            marker_pos = match.start()
            if marker_pos < start:
                shift_start += len(marker)
            if marker_pos < end:
                shift_end += len(marker)
    cleaned = text
    for marker in markers:
        cleaned = cleaned.replace(marker, "")
    return cleaned, max(0, start - shift_start), max(0, end - shift_end)


def check_unavailable(conv) -> bool:
    """Return True if any utterance in this conversation has been deleted/removed."""
    unavailable = {"[deleted]", "[removed]", "<INAUDIBLE>"}
    for utt in conv.iter_utterances():
        if utt.text in unavailable or utt.speaker.id in unavailable:
            return True
    return False


def build_reply_chain(utts_this_conv: list) -> dict:
    """Build the ancestry reply chain for every utterance in a conversation.

    Returns a dict mapping utterance_id → list of ancestor utterance ids
    (from direct parent up to root), mirroring the logic in do_reddit_search.
    """
    reply_chain_first_step = {u["id"]: u["reply_to"] for u in utts_this_conv}

    reply_chain = {}
    for k in reply_chain_first_step:
        reply_chain[k] = [reply_chain_first_step[k]]
        finished = False
        iteration = 0
        while not finished:
            iteration += 1
            if iteration >= 100:
                finished = True
                break
            if None in reply_chain[k]:
                finished = True
                break
            for other_k in reply_chain_first_step:
                if (
                    other_k != k
                    and other_k in reply_chain[k]
                    and reply_chain_first_step[other_k] not in reply_chain[k]
                ):
                    if reply_chain_first_step[other_k] is None:
                        finished = True
                        break
                    else:
                        reply_chain[k].append(reply_chain_first_step[other_k])

    # Patch utterances that ended up without an entry (edge case from original)
    for utt in utts_this_conv:
        if utt["id"] not in reply_chain:
            reply_chain[utt["id"]] = [None]

    return reply_chain


def get_thread_context(utts_this_conv: list, reply_chain: dict, utt_id: str) -> tuple:
    """Return (prev_subthread, current_utt, future_threads) for an annotated utterance.

    prev_subthread  — ancestor utterances ordered from root to direct parent
    current_utt     — the utterance dict for utt_id
    future_threads  — direct replies and their descendants (BFS order)
    """
    utt_by_id = {u["id"]: u for u in utts_this_conv}

    ancestor_ids = reply_chain.get(utt_id, [])
    prev_subthread = [
        utt_by_id[aid] for aid in ancestor_ids
        if aid is not None and aid in utt_by_id
    ]

    current_utt = utt_by_id[utt_id]

    # BFS through reply_chain to find all descendants
    queue = [
        k for k, ancestors in reply_chain.items()
        if k != utt_id and utt_id in ancestors and k in utt_by_id
    ]
    visited = set(queue)
    future_threads = [utt_by_id[k] for k in queue if k in utt_by_id]

    for _ in range(100):
        if not queue:
            break
        current = queue.pop(0)
        for next_k, ancestors in reply_chain.items():
            if next_k not in visited and current in ancestors:
                queue.append(next_k)
                visited.add(next_k)
                if next_k in utt_by_id:
                    future_threads.append(utt_by_id[next_k])

    return prev_subthread, current_utt, future_threads


def build_dialogue_html(
    utts_this_conv: list,
    reply_chain: dict,
    utt_id: str,
    start: int,
    end: int,
) -> tuple:
    """Reconstruct the Label Studio dialogue HTML for a single annotated item.

    Parameters
    ----------
    utts_this_conv : list of utterance dicts for the conversation
    reply_chain    : dict from build_reply_chain()
    utt_id         : ID of the utterance containing the quoted passage
    start, end     : character offsets of the quoted passage *within* the
                     utterance text (after modify_for_citation)

    Returns
    -------
    (html, html_quote_start, html_quote_end)
        html             — HTML string identical to the one produced by
                           do_reddit_search() during the original annotation
        html_quote_start — character offset of the quoted passage within html
                           (points just inside the blue highlight span)
        html_quote_end   — character offset of the end of the quoted passage
                           within html
    """
    # HTML templates (exact strings from original script)
    utt_intro  = (
        '<div style="clear: both">'
        '<div style="float: right; display: inline-block; '
        'border: 1px solid #F2F3F4; background-color: #F8F9F9; '
        'border-radius: 5px; padding: 7px; margin: 10px 0;">'
    )
    utt_outro  = "<p></p></div></div>"
    title_intro = (
        '<div style="display: inline-block; '
        'border: 1px solid #D5F5E3; background-color: #EAFAF1; '
        'border-radius: 5px; padding: 7px; margin: 10px 0;"><p>'
    )

    prev_subthread, current_utt, future_threads = get_thread_context(
        utts_this_conv, reply_chain, utt_id
    )

    dialogue_text = (
        title_intro + "<b>TITLE</b>: " + utts_this_conv[0]["text"] + utt_outro
    )

    for ps in prev_subthread:
        dialogue_text += utt_intro + "<b>" + ps["author"] + "</b>: " + ps["text"] + utt_outro

    delay_offset = len(dialogue_text) + len(
        utt_intro + "<b>" + current_utt["author"] + "</b>: "
    )

    dialogue_text += (
        utt_intro + "<b>" + current_utt["author"] + "</b>: "
        + current_utt["text"] + utt_outro
    )

    # Highlight the quoted passage in blue (exact style from original)
    highlight_open  = "<span style='color: #1924F7; font-weight: bold;'>"
    highlight_close = "</span>"
    html_quote_start = delay_offset + start
    html_quote_end   = delay_offset + end
    dialogue_text = (
        dialogue_text[: html_quote_start]
        + highlight_open
        + dialogue_text[html_quote_start : html_quote_end]
        + highlight_close
        + dialogue_text[html_quote_end :]
    )

    for ps in future_threads:
        dialogue_text += utt_intro + "<b>" + ps["author"] + "</b>: " + ps["text"] + utt_outro

    return dialogue_text, html_quote_start, html_quote_end


def build_plain_conversation(
    utts_this_conv: list,
    reply_chain: dict,
    utt_id: str,
    start: int,
    end: int,
) -> tuple:
    """Build a plain-text representation of the thread shown in Label Studio.

    Returns the same thread ordering as build_dialogue_html — title, ancestor
    utterances, the annotated utterance, then direct replies — but as a list
    of plain dicts instead of HTML.

    [STA-CITE] / [END-CITE] citation markers are stripped from all utterance
    texts, and the quote character offsets are adjusted accordingly.
    The positions are stored directly on the annotated utterance dict as
    ``quote_start`` / ``quote_end``.

    Parameters
    ----------
    utts_this_conv : list of utterance dicts for the conversation
    reply_chain    : dict from build_reply_chain()
    utt_id         : ID of the utterance containing the quoted passage
    start, end     : character offsets of the quoted passage within the
                     utterance text *with* citation markers (TSV values)

    Returns
    -------
    (conversation, annotated_utt_index)
        conversation        — ordered list of utterance dicts, each with:
                                utt_id   : convokit utterance ID
                                speaker  : Reddit username (plain)
                                text     : utterance text with citation
                                           markers stripped
                                reply_to : ID of the parent utterance,
                                           or null for the root post
              the annotated utterance additionally has:
                                is_annotated_utterance : True
                                quote_start : offset of quoted passage
                                              within this utterance's text
                                quote_end   : end offset (exclusive)
        annotated_utt_index — 0-based index of the annotated utterance
    """
    prev_subthread, current_utt, future_threads = get_thread_context(
        utts_this_conv, reply_chain, utt_id
    )

    def utt_to_plain(utt: dict, quote_start: int = None, quote_end: int = None) -> dict:
        cleaned_text = strip_citation_markers(utt["text"])
        d = {
            "utt_id":   utt["id"],
            "speaker":  utt.get("author_plain", utt["author"]),
            "text":     cleaned_text,
            "reply_to": utt["reply_to"],
        }
        if quote_start is not None:
            d["is_annotated_utterance"] = True
            d["quote_start"] = quote_start
            d["quote_end"]   = quote_end
        return d

    # Adjust quote offsets for the stripped annotated utterance
    _, plain_start, plain_end = strip_citation_markers(current_utt["text"], start, end)

    # Title pseudo-utterance
    title = utts_this_conv[0]
    conversation = [
        {"utt_id": title["id"], "speaker": "TITLE",
         "text": strip_citation_markers(title["text"]), "reply_to": None}
    ]

    for utt in prev_subthread:
        conversation.append(utt_to_plain(utt))

    annotated_utt_index = len(conversation)
    conversation.append(utt_to_plain(current_utt, plain_start, plain_end))

    for utt in future_threads:
        conversation.append(utt_to_plain(utt))

    return conversation, annotated_utt_index


# ---------------------------------------------------------------------------
# Load and pre-process conversations from WinningArguments corpus
# ---------------------------------------------------------------------------

def load_conversations(needed_conv_ids: set) -> dict:
    """Download/load WinningArguments and build utterance+reply structures.

    Returns a dict mapping conv_id → (utts_this_conv, reply_chain).
    Only conversations referenced in needed_conv_ids are fully processed.
    """
    print("Loading WinningArguments corpus via convokit …")
    print("(This will download the corpus on the first run, ~few hundred MB)")
    corpus = Corpus(filename=download("winning-args-corpus"))
    print(f"Corpus loaded.  Processing {len(needed_conv_ids)} conversations …")

    conversations = {}
    skipped = 0

    for conv in corpus.iter_conversations():
        conv_id = str(conv.id)
        if conv_id not in needed_conv_ids:
            continue

        if check_unavailable(conv):
            skipped += 1
            continue

        # Title as pseudo-utterance (same structure as original script)
        utts_this_conv = [
            {
                "id":       conv_id + "_title",
                "text":     conv.meta["op-title"],
                "author":   "TITLE",
                "reply_to": None,
            }
        ]

        for utt in conv.iter_utterances():
            text = modify_for_citation(utt.text)
            reply_to = getattr(utt, "reply_to", None)
            reply_txt = ("_##_rt-" + reply_to) if reply_to else ""
            utt_dict = {
                "id":           utt.id,
                "text":         text,
                "author":       utt.speaker.id + "_##_" + utt.id + reply_txt,
                "author_plain": utt.speaker.id,
                "reply_to":     reply_to,
            }
            utts_this_conv.append(utt_dict)

        reply_chain = build_reply_chain(utts_this_conv)
        conversations[conv_id] = (utts_this_conv, reply_chain)

    print(f"  Processed: {len(conversations)}  |  Skipped (unavailable): {skipped}")
    return conversations


# ---------------------------------------------------------------------------
# Main reconstruction logic
# ---------------------------------------------------------------------------

def main():
    # --- Load standoff annotations -----------------------------------------
    print(f"Reading standoff annotations from {INPUT_STANDOFF} …")
    standoff = json.load(open(INPUT_STANDOFF, encoding="utf-8"))
    print(f"  {len(standoff)} items.")

    # --- Load positional info TSV ------------------------------------------
    print(f"\nReading positional info from {INPUT_TSV} …")
    tsv = pd.read_csv(INPUT_TSV, sep="\t")
    # Build lookup: item_order → row (as dict)
    item_order_to_info = {
        int(row["item_order"]): row.to_dict()
        for _, row in tsv.iterrows()
    }
    print(f"  {len(tsv)} rows loaded.")

    # Verify all item_orders in standoff are present in TSV
    missing = [
        item["data"]["item_order"]
        for item in standoff
        if item["data"]["item_order"] not in item_order_to_info
    ]
    if missing:
        print(f"\nWARNING: {len(missing)} item_order values not found in TSV:")
        print("  ", missing[:10])

    # --- Identify needed conversations ------------------------------------
    needed_conv_ids = set()
    for item in standoff:
        needed_conv_ids.add(get_conv_id(item["data"]["id"]))
    print(f"\nUnique conversations referenced: {len(needed_conv_ids)}")

    # --- Load and process WinningArguments ---------------------------------
    conversations = load_conversations(needed_conv_ids)

    # --- Reconstruct each item --------------------------------------------
    print("\nReconstructing dialogue HTML for each item …")
    reconstructed = []
    errors = []

    for item in standoff:
        item_order = item["data"]["id"]
        conv_id    = get_conv_id(item["data"]["id"])
        utt_id     = "_".join(item["data"]["id"].split("_")[2:-1])
        order_key  = item["data"]["item_order"]

        if conv_id not in conversations:
            errors.append(f"Conversation not found: {conv_id}  (item {item_order})")
            continue

        info = item_order_to_info.get(order_key)
        if info is None:
            errors.append(f"TSV row not found for item_order={order_key}  (item {item_order})")
            continue

        utts_this_conv, reply_chain = conversations[conv_id]

        try:
            dialogue_html, html_qs, html_qe = build_dialogue_html(
                utts_this_conv=utts_this_conv,
                reply_chain=reply_chain,
                utt_id=utt_id,
                start=int(info["start"]),
                end=int(info["end"]),
            )
            conversation, ann_idx = build_plain_conversation(
                utts_this_conv=utts_this_conv,
                reply_chain=reply_chain,
                utt_id=utt_id,
                start=int(info["start"]),
                end=int(info["end"]),
            )
        except Exception as e:
            errors.append(f"Build failed for {item_order}: {e}")
            continue

        # Assemble the full item
        full_item = {
            "data": {
                # --- Label Studio HTML representation ---
                "html_conversation": dialogue_html,
                "html_quote_start": html_qs,
                "html_quote_end":   html_qe,
                # --- Plain-text conversation representation ---
                # start / end are offsets within conversation[annotated_utt_index]["text"]
                "conversation":          conversation,
                "annotated_utt_index":   ann_idx,
                # --- Shared fields ---
                "id":                           item["data"]["id"],
                "quoted_passage":               item["data"]["quoted_passage"],
                "item_order":                   item["data"]["item_order"],
                "used_previously_in_subthread": item["data"]["used_previously_in_subthread"],
                "sign":  info["sign"],
                "ngram": info["ngram"],
                "start": int(info["start"]),
                "end":   int(info["end"]),
            },
            "annotations": item["annotations"],
        }
        reconstructed.append(full_item)

    # --- Report and save ---------------------------------------------------
    print(f"  Successfully reconstructed: {len(reconstructed)}")
    if errors:
        print(f"  Errors ({len(errors)}):")
        for e in errors[:20]:
            print(f"    {e}")

    print(f"\nWriting reconstructed annotations to {OUTPUT_JSON} …")
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(reconstructed, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("RECONSTRUCTION COMPLETE")
    print("=" * 60)
    print(f"  Input items:         {len(standoff)}")
    print(f"  Reconstructed:       {len(reconstructed)}")
    print(f"  Errors / skipped:    {len(standoff) - len(reconstructed)}")
    print(f"  Output:              {OUTPUT_JSON}")
    print("=" * 60)
    print("\nEach item contains two conversation representations:")
    print("  data.dialogue / html_quote_start / html_quote_end")
    print("      → Label Studio HTML + character offsets within HTML")
    print("  data.conversation / annotated_utt_index")
    print("      → Plain-text thread (citation markers stripped);")
    print("        conversation[annotated_utt_index] has quote_start / quote_end")
    print("  data.sign, data.ngram, and all annotation labels are also present.")


if __name__ == "__main__":
    main()
