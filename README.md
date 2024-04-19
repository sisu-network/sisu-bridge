# sisu-bridge

This is an implementation for a [decentralized trustless bridge](https://arxiv.org/abs/2404.10404) paper. This is a prototype, not a full implementation of the Sisu protocol. You are welcomed to modify and use the code but it's not ready for production. 

You will need a NVIDIA graphic card and [nvidia-smi](https://developer.nvidia.com/nvidia-system-management-interface) command tools installed to run. 

The `src/main.rs` contains various tests for public key aggregation, Merkle tree proof, VPD, etc. The `groth16` folder contains code for distributed MSM computation in groth16. `sha256-circuit` and `pubkey-aggregation` are optimized circuit for public key SHA256 & pubkey aggregation in Ethereum consensus. `bls-circom` consists of verification program written in Circom language to verify the Phase 1 of Sisu. 

We test on a machine Intel(R) 8 cores i7-7700 @3.60GHz CPU, 32GB RAM and Geforce GTX 1080 (8GB VRAM) GPU. This is a modest hardware configuration. One Geforce GTX 1080 can generate proof for 16 validators. Fully optimized code can handle 32-64 validators as many of the code is not moved to GPU yet. 
