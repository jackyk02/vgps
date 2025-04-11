from .continuous.bc import BCAgent
from .continuous.calql import CalQLAgent
from .continuous.cql import ContinuousCQLAgent
from .continuous.gc_bc import GCBCAgent
from .continuous.gc_ddpm_bc import GCDDPMBCAgent
from .continuous.gc_iql import GCIQLAgent
from .continuous.iql import IQLAgent
from .continuous.sac import SACAgent

agents = {
    "gc_bc": GCBCAgent,
    "gc_iql": GCIQLAgent,
    "gc_ddpm_bc": GCDDPMBCAgent,
    "bc": BCAgent,
    "iql": IQLAgent,
    "cql": ContinuousCQLAgent,
    "calql": CalQLAgent,
    "sac": SACAgent,
}
