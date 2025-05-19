# Projects

<script setup>
import ResearchProject from './.vitepress/theme/components/ResearchProject.vue'
</script>

## Open source software

<ResearchProject
    title="Piccolo.jl"
    img="/images/piccolo.jpg"
    link="https://github.com/harmoniqs/Piccolo.jl">
    Fine tuned quantum control inspired by robotics, offerred by Harmoniqs and written in Julia.
</ResearchProject>

[**Harmoniqs**](https://www.harmoniqs.co/) is a startup and _open-source ecosystem_ that I co-founded because quantum computing scientists and engineers should have easy access to state-of-the-art optimal control and calibration solutions.

## Research projects

My research focuses on novel applications of optimal control for quantum computing, fast automated calibration of control pulses, learning models for quantum devices, and new methods for optimal control of quantum systems.

I am currently interested in accelerated computing for quantum control (GPUs, HPC), and bosonic quantum error correction.

<ResearchProject
    title="Crosstalk-Robust Gate Sets"
    img="/images/crgs.png">
    Globally suppress crosstalk on a quantum computer using orthogonal gate sets.
</ResearchProject>

<ResearchProject
    title="Quantum Iterative Learning Control"
    img="/images/qilc.png">
    Highly-efficient, model-based calibration of quantum optimal control pulses.
</ResearchProject>

<ResearchProject
    title="Quantum Trajectory Bundles"
    img="/images/bundles.jpg">
    Massively-parallel, derivative-free quantum optimal control.
</ResearchProject>

<ResearchProject
    title="Optimal control and neural networks can efficiently interpolate gates"
    link="https://www.computer.org/csdl/proceedings-article/qce/2024/413701b336/23oq4Nyibuw"
    img="/images/interp.png">
    Combine a coordinated optimal control problem with neural networks to efficiently synthesize (and calibrate) pulses that interpolate among continously-parameterized gate families.
</ResearchProject>

<ResearchProject
    title="Robust quantum state prep with MPC"
    link="https://quantum-journal.org/papers/q-2022-10-13-837/"
    img="/images/mpc.png">
    Model predictive control (MPC) enables robust quantum operations using device feedback.
</ResearchProject>

<ResearchProject
    title="Learn quantum control models from data with BiDMD"
    link="https://iopscience.iop.org/article/10.1088/1367-2630/abe972"
    img="/images/bidmd.png">
    Build control models of quantum devices directly from data via a bilinear dynamic mode decomposition (DMD), and extend the idea to stroboscopic measurements using Floquet theory.
</ResearchProject>
