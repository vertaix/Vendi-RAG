"""
A synthetic multi-hop corpus, and the metrics to score retrieval on it.

The corpus is a small fictional world — Arctic research expeditions, the people
who led them, the instruments they carried, the observatories that house those
instruments, and the towns those observatories sit above.  Answering a question
like

    "In which town is the observatory that houses the instrument used by the
     leader of the Thule Expedition?"

requires chaining four separate facts, each stated in a different document, and
each phrased in the vocabulary of its *own* hop rather than the question's.

Around every one of those chains the generator plants a dense thicket of
near-duplicate documents that repeat the expedition's name — schedules, funding
notes, press coverage — the way a real corpus accumulates redundant coverage of
whatever entity the question happens to name.  Similarity search fills its
entire budget with that thicket, because those documents look most like the
question.  Everything after the first hop is missed.

That is the failure mode Vendi retrieval targets, and this module exists so it
can be measured rather than asserted:

    >>> from vendirag.toy import make_corpus, hop_coverage
    >>> corpus = make_corpus(n_chains=20, seed=0)
    >>> len(corpus.documents), len(corpus.questions)
    (320, 60)

Every document carries ``metadata["role"]`` (``"hop"`` or ``"distractor"``) and
``metadata["chain"]``, so retrieval can be scored exactly.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from .types import Document

__all__ = ["ToyQuestion", "ToyCorpus", "make_corpus", "hop_coverage", "evaluate_retrieval"]


def _cap(text: str) -> str:
    """Capitalize the first letter only — ``str.capitalize`` would lowercase
    the proper nouns that make up most of these sentences."""
    return text[:1].upper() + text[1:]


# ── the name bank: enough parts to build unique, plausible entities ─────────

_EXPEDITION_A = [
    "Thule", "Kittiwake", "Meridian", "Halvard", "Orsted", "Bellwether",
    "Nansen", "Fjord", "Larkspur", "Cormorant", "Vantage", "Sable",
    "Erebus", "Willow", "Ptarmigan", "Auger", "Lodestar", "Kestrel",
    "Bramble", "Sedge", "Tarn", "Quillan", "Marrow", "Ashgate",
    "Pellworm", "Rimfrost", "Skerry", "Ivory", "Dovekie", "Grindle",
    "Nautilus", "Corbel", "Windlass", "Fathom", "Alder", "Beacon",
    "Cinder", "Drift", "Ember", "Frost",
]
_EXPEDITION_B = ["Expedition", "Traverse", "Survey", "Crossing"]

_FIRST = [
    "Mira", "Anselm", "Yusra", "Tobias", "Ingrid", "Rafael", "Noor", "Estelle",
    "Kwame", "Solveig", "Hideo", "Camille", "Dmitri", "Farah", "Lucian",
    "Marit", "Osric", "Priya", "Quentin", "Rhoda", "Sacha", "Tamsin",
    "Ulf", "Vesna", "Wren", "Xiulan", "Yannick", "Zora", "Aurel", "Brigid",
    "Corin", "Delphine", "Emeric", "Freya", "Gustav", "Halina", "Ivo",
    "Juno", "Kaspar", "Liesel",
]
_LAST = [
    "Kovac", "Aldritch", "Benali", "Carrow", "Delaine", "Ebersole", "Fenwick",
    "Grigore", "Halloran", "Ingemar", "Jarvela", "Kestenbaum", "Lindqvist",
    "Mbeki", "Nordahl", "Okonkwo", "Pashkov", "Quandt", "Reinholt", "Serrano",
    "Thackery", "Ulvestad", "Vasquez", "Wexley", "Ymir", "Zabala", "Ashford",
    "Brandt", "Chevalier", "Dunmore", "Elstrom", "Faraday", "Guthrie",
    "Hedlund", "Ivarsson", "Joubert", "Kranz", "Lamotte", "Meinhardt", "Novak",
]

_INSTRUMENT_A = [
    "Halcyon", "Meridian", "Cassiopeia", "Lumen", "Perihelion", "Quartzline",
    "Solstice", "Vireo", "Zenith", "Aperture", "Basalt", "Coronal", "Duskline",
    "Equinox", "Fulcrum", "Gyral", "Helios", "Isotherm", "Jovian", "Kelvin",
    "Latitude", "Magnetite", "Nadir", "Obsidian", "Parallax", "Quadrant",
    "Radiant", "Sextant", "Tessellate", "Umbra", "Vector", "Wavefront",
    "Xenolith", "Yardarm", "Zircon", "Alluvial", "Bathyal", "Cryostat",
    "Diopter", "Ecliptic",
]
_INSTRUMENT_B = [
    "Interferometer", "Spectrograph", "Magnetometer", "Radiometer",
    "Gravimeter", "Photometer",
]

_OBSERVATORY_A = [
    "Rowan", "Petrel", "Blackwood", "Sundal", "Corvid", "Ashvale", "Wintermere",
    "Glenmara", "Torrance", "Highfell", "Marrowgate", "Silverbeck", "Northrop",
    "Falkenrath", "Elderbrook", "Cairnross", "Dunmoor", "Bellhaven",
    "Larkmount", "Stonefield", "Ravensholm", "Ironbrand", "Thistlewood",
    "Oakenshaw", "Pinewatch", "Quarrymoor", "Redfern", "Saltmarsh", "Thornbury",
    "Underhill", "Vaultridge", "Westmere", "Yarrowfield", "Zephyr", "Amberlyn",
    "Brackwater", "Coldharbour", "Dovecote", "Everstone", "Fernhollow",
]

_TOWN = [
    "Yellowknife", "Kirkenes", "Longyear", "Narsaq", "Ilulissat", "Hammerfest",
    "Tiksi", "Iqaluit", "Uummannaq", "Vardo", "Pevek", "Qaanaaq", "Barentsburg",
    "Nuuk", "Inuvik", "Anadyr", "Sisimiut", "Alta", "Bodo", "Chokurdakh",
    "Dikson", "Egedesminde", "Fort Ross", "Gjoa Haven", "Honningsvag",
    "Igarka", "Jokkmokk", "Kotzebue", "Leknes", "Murmansk", "Nome", "Olenek",
    "Pangnirtung", "Qikiqtarjuaq", "Resolute", "Svolvaer", "Tromso", "Utqiagvik",
    "Verkhoyansk", "Wainwright",
]

_FIELD = [
    "glaciology", "atmospheric physics", "sea-ice dynamics", "paleoclimatology",
    "geomagnetism", "auroral science", "permafrost hydrology", "radio astronomy",
]

# One funder per chain.  Every proper noun in the corpus belongs to exactly one
# chain, so an entity-graph walk cannot leak between chains through a shared hub
# — the corpus tests retrieval, not entity disambiguation.
_FUNDER_A = [
    "Verrell", "Coastal", "Nordheim", "Arbor", "Fairweather", "Lindhardt",
    "Marchetti", "Sunderland", "Okoye", "Brightwater", "Calloway", "Deverell",
    "Emberton", "Fairholm", "Grieveson", "Hartnell", "Illingworth", "Jessop",
    "Kingsmill", "Lavelle", "Merriwether", "Norbury", "Ollerenshaw", "Prentiss",
    "Quilter", "Rutherglen", "Standish", "Trelawney", "Uxbridge", "Vandermeer",
    "Wetherby", "Yarborough", "Zelinsky", "Ainsworth", "Balfour", "Cadwallader",
    "Drummond", "Ellingham", "Fotheringay", "Glenister",
]
_FUNDER_B = ["Foundation", "Trust", "Endowment", "Fund", "Bequest"]


@dataclass
class ToyQuestion:
    """One multi-hop question with its full gold evidence chain."""

    question: str
    answer: str
    #: Document ids that must all be retrieved for the chain to be answerable.
    gold_ids: List[str]
    n_hops: int
    chain: int

    def coverage(self, documents: Sequence) -> float:
        """Fraction of this question's gold documents present in ``documents``."""
        got = {getattr(d, "id", None) for d in documents}
        if not self.gold_ids:
            return 0.0
        return sum(1 for g in self.gold_ids if g in got) / len(self.gold_ids)


@dataclass
class ToyCorpus:
    documents: List[Document]
    questions: List[ToyQuestion]
    #: chain index -> the entities used to build it, for narration in demos.
    entities: List[dict] = field(default_factory=list)

    @property
    def texts(self) -> List[str]:
        return [d.text for d in self.documents]

    def by_id(self, doc_id: str) -> Document:
        return next(d for d in self.documents if d.id == doc_id)

    def describe(self) -> str:
        n_hop = sum(1 for d in self.documents if d.metadata.get("role") == "hop")
        return (
            f"{len(self.documents)} documents "
            f"({n_hop} evidence, {len(self.documents) - n_hop} near-duplicate distractors), "
            f"{len(self.questions)} questions across "
            f"{sorted({q.n_hops for q in self.questions})} hops"
        )


def make_corpus(n_chains: int = 20, seed: int = 0, n_distractors: int = 12) -> ToyCorpus:
    """Generate the synthetic corpus.

    Parameters
    ----------
    n_chains : int
        Number of independent 4-fact chains (max 40).  Each contributes three
        questions (2-hop, 3-hop, 4-hop) and ``4 + n_distractors`` documents.
    seed : int
        Seeds the distractor shuffling; the entities themselves are assigned
        deterministically so a given ``n_chains`` always yields the same world.
    n_distractors : int
        Redundant documents planted around each chain's head entity, up to 28.
        This is the depth of the thicket relevance-only retrieval has to see
        past; sweeping it shows where similarity search breaks.
    """
    if not 1 <= n_chains <= len(_EXPEDITION_A):
        raise ValueError(f"n_chains must be between 1 and {len(_EXPEDITION_A)}")
    rng = random.Random(seed)

    documents: List[Document] = []
    questions: List[ToyQuestion] = []
    entities: List[dict] = []

    for c in range(n_chains):
        exp = f"the {_EXPEDITION_A[c]} {_EXPEDITION_B[c % len(_EXPEDITION_B)]}"
        exp_short = f"{_EXPEDITION_A[c]} {_EXPEDITION_B[c % len(_EXPEDITION_B)]}"
        leader = f"Dr. {_FIRST[c]} {_LAST[c]}"
        instrument = f"the {_INSTRUMENT_A[c]} {_INSTRUMENT_B[c % len(_INSTRUMENT_B)]}"
        instrument_short = f"{_INSTRUMENT_A[c]} {_INSTRUMENT_B[c % len(_INSTRUMENT_B)]}"
        observatory = f"the {_OBSERVATORY_A[c]} Observatory"
        observatory_short = f"{_OBSERVATORY_A[c]} Observatory"
        town = _TOWN[c]
        field_ = _FIELD[c % len(_FIELD)]
        funder = f"the {_FUNDER_A[c]} {_FUNDER_B[c % len(_FUNDER_B)]}"
        year = 1962 + (c * 3) % 45

        ent = dict(
            expedition=exp_short, leader=leader, instrument=instrument_short,
            observatory=observatory_short, town=town, chain=c,
        )
        entities.append(ent)

        # ── the evidence chain: one fact per document, each in its own idiom ──
        hops = [
            (f"c{c}_hop1",
             f"{_cap(exp)} was led by {leader}, a specialist in {field_} "
             f"who had joined the programme four years earlier."),
            (f"c{c}_hop2",
             f"{leader} recorded every reading with {instrument}, the instrument "
             f"that the team relied on throughout the campaign."),
            (f"c{c}_hop3",
             f"{_cap(instrument)} is housed at {observatory}, the observatory "
             f"that has maintained it since it was commissioned."),
            (f"c{c}_hop4",
             f"{_cap(observatory)} stands on a ridge above {town}, the town "
             f"whose harbour supplies its crews each season."),
        ]
        for hop_i, (doc_id, text) in enumerate(hops, start=1):
            documents.append(Document(
                text=text, id=doc_id,
                metadata={"role": "hop", "chain": c, "hop": hop_i},
            ))

        # ── the redundancy trap ──────────────────────────────────────────────
        # Duplicate coverage, the way a real corpus accumulates it around a
        # notable entity: a handful of facts restated over and over, every
        # restatement saturated with the expedition's name.  Two of the four
        # families are deliberately adjacent to what the questions ask —
        # leadership and equipment — without ever naming the leader or the
        # instrument, so they compete with the real evidence on topic and not
        # just on surface form.  Relevance-only retrieval fills its whole
        # budget from here.
        bases = [
            f"{exp} departed in the spring of {year} and returned eleven "
            f"months later.",
            f"{exp} was financed by {funder}, which covered the full cost of "
            f"the campaign.",
            f"the leadership of {exp} rotated between two watches through the "
            f"winter months.",
            f"the equipment carried on {exp} travelled north in numbered "
            f"wooden crates.",
        ]
        prefixes = [
            "", "Records show that ", "According to the published chronology, ",
            "Contemporary reports state that ", "Archival sources confirm that ",
            "The official account notes that ", "It is well documented that ",
        ]
        trap = [
            (_cap(base) if not prefix else prefix + base, family)
            for family, base in enumerate(bases)
            for prefix in prefixes
        ]
        rng.shuffle(trap)
        for j, (text, family) in enumerate(trap[:n_distractors]):
            documents.append(Document(
                text=text, id=f"c{c}_d{j}",
                metadata={"role": "distractor", "chain": c, "family": family},
            ))

        # ── questions: 2, 3, and 4 hops over the same chain ───────────────────
        questions.append(ToyQuestion(
            question=f"Which instrument was used by the leader of {exp}?",
            answer=instrument_short,
            gold_ids=[f"c{c}_hop1", f"c{c}_hop2"],
            n_hops=2, chain=c,
        ))
        questions.append(ToyQuestion(
            question=(
                f"At which observatory is the instrument used by the leader of "
                f"{exp} housed?"
            ),
            answer=observatory_short,
            gold_ids=[f"c{c}_hop1", f"c{c}_hop2", f"c{c}_hop3"],
            n_hops=3, chain=c,
        ))
        questions.append(ToyQuestion(
            question=(
                f"In which town is the observatory that houses the instrument "
                f"used by the leader of {exp}?"
            ),
            answer=town,
            gold_ids=[f"c{c}_hop1", f"c{c}_hop2", f"c{c}_hop3", f"c{c}_hop4"],
            n_hops=4, chain=c,
        ))

    return ToyCorpus(documents=documents, questions=questions, entities=entities)


# ── metrics ─────────────────────────────────────────────────────────────────

def hop_coverage(question: ToyQuestion, documents: Sequence) -> float:
    """Fraction of ``question``'s gold evidence present in ``documents``."""
    return question.coverage(documents)


def evaluate_retrieval(
    questions: Sequence[ToyQuestion],
    retrieve: Callable[[str], Sequence],
    label: str = "",
) -> Dict[str, float]:
    """Score a retrieval function on the toy questions.

    ``retrieve`` takes a question string and returns documents.

    Returns
    -------
    dict with

    ``hop_coverage``
        Mean fraction of gold evidence retrieved — partial credit per hop.
    ``answerable``
        Fraction of questions where *every* hop was retrieved.  This is the one
        that matters: a chain with one missing link cannot be answered.
    ``redundancy``
        Mean number of retrieved documents per *distinct* gold chain touched —
        a proxy for wasted context budget.
    """
    coverages, complete, wasted = [], 0, []
    for q in questions:
        docs = list(retrieve(q.question))
        cov = q.coverage(docs)
        coverages.append(cov)
        complete += int(cov >= 1.0)
        n_distract = sum(1 for d in docs if getattr(d, "metadata", {}).get("role") != "hop")
        wasted.append(n_distract / max(len(docs), 1))
    n = max(len(questions), 1)
    out = {
        "hop_coverage": sum(coverages) / n,
        "answerable": complete / n,
        "distractor_rate": sum(wasted) / n,
    }
    if label:
        out["label"] = label  # type: ignore[assignment]
    return out
