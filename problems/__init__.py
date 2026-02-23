from problems.base import NO_WINDOWING, Problem, ProblemSpec
from problems.brax_ant import BraxAntSpec
from problems.hpo import HPOSpec
from problems.sequence_copy import SequenceCopySpec

problems: list[ProblemSpec] = [
    HPOSpec(),
    BraxAntSpec(),
    SequenceCopySpec(),
    SequenceCopySpec(linear=True),
]
