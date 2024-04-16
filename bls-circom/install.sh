#!/bin/bash

update_essential() {
  sudo apt-get update -y
  sudo apt-get install build-essential libgmp-dev libsodium-dev nasm nlohmann-json3-dev libssl-dev \
  zlib1g zlib1g-dev libbz2-dev libomp-dev cmake m4 ca-certificates -y
}

clone_repo() {
  chmod 600 /home/ubuntu/.ssh/id_ed25519
  cd ~
  mkdir code
  cd code
  yes Y | git clone git@github.com:questx-lab/zkp-practice.git
  cd zkp-practice
  git checkout feat/libra
}

install_rust() {
  echo 1 | curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  source ~/.bashrc
}

install_snarkjs() {
  cd ~
  curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
  export NVM_DIR="$([ -z "${XDG_CONFIG_HOME-}" ] && printf %s "${HOME}/.nvm" || printf %s "${XDG_CONFIG_HOME}/nvm")"
  [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh" # This loads nvm
  source ~/.bashrc
  nvm install v14.8.0
  nvm use v14.8.0
  npm install -g snarkjs@latest
  source ~/.bashrc

  cd ~/code
  git clone https://github.com/iden3/snarkjs.git
  cd snarkjs
  npm i
}

install_circom() {
  cd ~
  cd code
  git clone https://github.com/iden3/circom.git
  cd circom
  cargo build --release
  cargo install --path circom
  cd ..
  source ~/.bashrc
}

download_tau() {
  cd ~
  mkdir -p code/zkp-practice/anonnice1999/bls-circom/common
  mkdir -p tau
  cd tau
  wget https://storage.googleapis.com/zkevm/ptau/powersOfTau28_hez_final_18.ptau
  cd ~
  cp ./tau/powersOfTau28_hez_final_18.ptau code/zkp-practice/anonnice1999/bls-circom/common/pot12_final.ptau
}

install_python() {
  cd ~
  curl https://pyenv.run | bash
  echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
  echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
  echo 'eval "$(pyenv init -)"' >> ~/.bashrc
  echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
  source ~/.bashrc
  CFLAGS=-I/usr/include/openssl LDFLAGS=-L/usr/lib pyenv install -v 3.8.16
  pyenv global 3.8.16
  source ~/.bashrc
}

patch_node() {
  sourece ~/.bashrc
  nvm install v14.8.0
  cd ~/code
  git clone https://github.com/nodejs/node.git
  cd node
  git checkout 8beef5eeb82425b13d447b50beafb04ece7f91b1
  patch -p1 <<EOL
index 0097683120..d35fd6e68d 100644
--- a/deps/v8/src/api/api.cc
+++ b/deps/v8/src/api/api.cc
@@ -7986,7 +7986,7 @@ void BigInt::ToWordsArray(int* sign_bit, int* word_count,
 void Isolate::ReportExternalAllocationLimitReached() {
   i::Heap* heap = reinterpret_cast<i::Isolate*>(this)->heap();
   if (heap->gc_state() != i::Heap::NOT_IN_GC) return;
-  heap->ReportExternalMemoryPressure();
+  // heap->ReportExternalMemoryPressure();
 }

 HeapProfiler* Isolate::GetHeapProfiler() {
diff --git a/deps/v8/src/objects/backing-store.cc b/deps/v8/src/objects/backing-store.cc
index bd9f39b7d3..c7d7e58ef3 100644
--- a/deps/v8/src/objects/backing-store.cc
+++ b/deps/v8/src/objects/backing-store.cc
@@ -34,7 +34,7 @@ constexpr bool kUseGuardRegions = false;
 // address space limits needs to be smaller.
 constexpr size_t kAddressSpaceLimit = 0x8000000000L;  // 512 GiB
 #elif V8_TARGET_ARCH_64_BIT
-constexpr size_t kAddressSpaceLimit = 0x10100000000L;  // 1 TiB + 4 GiB
+constexpr size_t kAddressSpaceLimit = 0x40100000000L;  // 4 TiB + 4 GiB
 #else
 constexpr size_t kAddressSpaceLimit = 0xC0000000;  // 3 GiB
 #endif
EOL
  ./configure
  make -j16
}

install_rapid_snark() {
  cd ~/code
  if [ ! -d "rapidsnark" ]; then
    echo "No rapidsnark folder found. Cloning from github..."
    git clone git@github.com:iden3/rapidsnark.git
  fi
  cd rapidsnark
  git submodule init
  git submodule update
  ./build_gmp.sh host
  mkdir build_prover && cd build_prover
  echo pwd = $pwd
  cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../package
  make -j4 && make install
}

setup_swap() {
  sudo fallocate -l 400G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile

  sudo sh -c 'echo "vm.max_map_count=10000000" > /etc/sysctl.conf'
  sudo sh -c 'echo 10000000 > /proc/sys/vm/max_map_count'
}

update_essential
# clone_repo
# install_rust
install_snarkjs
patch_node
install_python
install_circom
install_rapid_snark
# download_tau
# setup_swap
