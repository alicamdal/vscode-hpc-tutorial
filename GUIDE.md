# Tutorial guide: VS Code, the UT HPC cluster, and a Vision Transformer

**Course:** GenAI Models and Robotic Applications (2026-1B)
**Tutorial date:** 20 November 2026

This guide takes you from an empty laptop to two working programs:

1. a small **GPU job on the UT HPC cluster** that prints the PyTorch version, the GPU name, and its memory;
2. a **Vision Transformer (ViT) on your own laptop** that shows how an image becomes patches, vectors, attention, and a prediction.

Steps are given for **Windows** and **Ubuntu**. Work through the parts in order. Each step ends with a
**Check** so you know it worked before you move on.

> **macOS users:** follow the Ubuntu steps in the macOS Terminal. Install Python from
> [python.org](https://www.python.org/downloads/) and Git by running `xcode-select --install`.
> Ask the TA if something looks different.

## Contents

- [Part 0. Before you start](#part-0-before-you-start)
- [Part A. Install the tools on your laptop](#part-a-install-the-tools-on-your-laptop)
- [Part B. Connect to the HPC cluster](#part-b-connect-to-the-hpc-cluster)
- [Part C. Run a GPU job on the HPC cluster](#part-c-run-a-gpu-job-on-the-hpc-cluster)
- [Part D. Run the Vision Transformer on your laptop](#part-d-run-the-vision-transformer-on-your-laptop)
- [Part E. (Optional) An AI coding assistant](#part-e-optional-an-ai-coding-assistant)
- [Troubleshooting](#troubleshooting)

### How to read the commands

- Lines in grey boxes are commands. Type or paste them into a terminal and press Enter.
- Replace `s1234567` with **your own** UT username (your student number with the `s`, lowercase).
- When a box is labelled **Windows (PowerShell)** or **Ubuntu**, use only the one for your system.

---

## Part 0. Before you start

You need:

- [ ] your UT username and password (the same as for Canvas)
- [ ] access to the HPC cluster for this course (your teacher arranges this)
- [ ] a laptop with about **5 GB of free disk space** and permission to install software
- [ ] (optional, for Part E) a GitHub account

---

## Part A. Install the tools on your laptop

### A1. Visual Studio Code

VS Code is the editor we use. It can also open folders on a remote server, which we need in Part B.

**Windows**: download the *User Installer* from <https://code.visualstudio.com/download> and run it.
On the "Select Additional Tasks" page, tick **Add to PATH**.

**Ubuntu**: download the `.deb` file from <https://code.visualstudio.com/download>, then:

```bash
sudo apt install ./Downloads/code_*.deb
```

**Check:** open VS Code. You see the Welcome page.

### A2. Two VS Code extensions

In VS Code, click the **Extensions** icon on the left (four small squares) or press `Ctrl+Shift+X`.
Search for and install:

1. **Python** (publisher: Microsoft)
2. **Remote - SSH** (publisher: Microsoft)

**Check:** a small green or blue `><` button appears in the bottom-left corner of VS Code.

### A3. Git

Git downloads ("clones") code from GitHub.

**Windows**: download Git from <https://git-scm.com/download/win> and run the installer. The default
options are fine.

**Ubuntu**:

```bash
sudo apt update
sudo apt install git
```

**Check:** open a **new** terminal (in VS Code: menu *Terminal > New Terminal*) and run:

```bash
git --version
```

You see something like `git version 2.43.0`.

### A4. Python

PyTorch needs **Python 3.11 or newer**. We recommend **Python 3.12**.

**Windows**: download Python 3.12 from <https://www.python.org/downloads/windows/>. In the first
installer screen, tick **"Add python.exe to PATH"**, then click *Install Now*.

**Ubuntu 24.04** already has Python 3.12. Add the tools for virtual environments:

```bash
sudo apt install python3-venv python3-pip
```

**Check:** in a **new** terminal:

| Windows (PowerShell) | Ubuntu |
|---|---|
| `py --version` | `python3 --version` |

You see `Python 3.12.x` (or 3.11, 3.13).

### A5. eduVPN

The HPC cluster only accepts connections from the UT network. On campus (eduroam) you are already
inside. **At home you need eduVPN.** Install it now, so you can use the cluster after the tutorial too.

**Windows**:

1. Go to <https://ut.eduvpn.nl>, log in with your UT account, and click **Windows** to download the app.
2. Install it and open **eduVPN** from the Start menu.
3. Under **Institute Access**, search for **University of Twente** and select it.
4. Log in with your UT account in the browser window that opens, and click **Approve**.

**Ubuntu**: the eduVPN team provides an install script
([official instructions](https://docs.eduvpn.org/client/linux/installation.html)):

```bash
curl --proto '=https' --tlsv1.2 https://docs.eduvpn.org/client/linux/install.sh -O
bash ./install.sh
```

Then open **eduVPN** from the applications menu and follow steps 3 and 4 above.

**Check:** the eduVPN app shows **Connected**. (On campus you can disconnect it again.)

UT help page: <https://www.utwente.nl/en/service-portal/hardware-software-network/network-eduroam-vpn-etc/eduvpn-connection>

---

## Part B. Connect to the HPC cluster

### What is the HPC cluster?

The cluster is many computers that work together. You never work on the GPU computers directly.

```
 your laptop  ──ssh──►  login node (hpc-head1)  ──sbatch──►  compute node with GPUs
  (VS Code)              edit files, install,                 runs your job,
                         submit jobs. NO heavy work!          writes the output file
```

- The **login node** is shared by everyone. Use it to edit code, install packages, and submit jobs.
  **Never run training or other heavy programs on it.**
- The **compute nodes** have the GPUs. You reach them by submitting a **job** to the scheduler (**Slurm**).

### B1. First connection from a terminal

Open a terminal (Windows: *PowerShell*; Ubuntu: *Terminal*) and run:

```bash
ssh s1234567@hpc-head1.ewi.utwente.nl
```

- The first time, ssh asks `Are you sure you want to continue connecting?`. Type `yes`.
- Type your UT password. **Nothing appears while you type.** This is normal. Press Enter.

If you see `Permission denied (keyboard-interactive)`, use this command instead:

```bash
ssh -o PreferredAuthentications=password s1234567@hpc-head1.ewi.utwente.nl
```

**Check:** the prompt changes to something like `s1234567@hpc-head1:~$`. Type `exit` to log out.

### B2. Save the connection in an SSH config file

This file lets VS Code (and you) connect with a short name.

**Windows (PowerShell)**:

```powershell
mkdir -Force $HOME\.ssh
notepad $HOME\.ssh\config
```

**Ubuntu**:

```bash
mkdir -p ~/.ssh
nano ~/.ssh/config
```

Paste these lines, put in your own username, and save the file
(Notepad: *File > Save*; when it asks, keep the name `config` **without** `.txt`. nano: `Ctrl+O`, Enter, `Ctrl+X`):

```
Host ut-hpc
    HostName hpc-head1.ewi.utwente.nl
    User s1234567
    PreferredAuthentications publickey,password
```

**Check:** `ssh ut-hpc` now logs you in. Type `exit` to log out.

### B3. (Recommended) An SSH key, so you do not type your password every time

```bash
ssh-keygen -t ed25519
```

Press Enter three times (default file, no passphrase). Then copy the public key to the cluster:

**Windows (PowerShell)**:

```powershell
type $HOME\.ssh\id_ed25519.pub | ssh ut-hpc "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
```

**Ubuntu**:

```bash
ssh-copy-id ut-hpc
```

Type your password one last time.

**Check:** `ssh ut-hpc` logs you in **without** asking for a password.

### B4. Connect from VS Code

1. Click the `><` button in the bottom-left corner of VS Code.
2. Choose **Connect to Host...** and then **ut-hpc**.
3. If asked for the platform, choose **Linux**. Type your password if asked.
4. The first connection takes about a minute: VS Code installs a small helper on the cluster.

**Check:** the bottom-left corner shows **SSH: ut-hpc**. Open a terminal (*Terminal > New Terminal*).
It is a terminal **on the cluster**: run `hostname` and you see `hpc-head1`.

---

## Part C. Run a GPU job on the HPC cluster

All commands in Part C run in the VS Code terminal that is connected to **ut-hpc**.

### C1. Clone the project

```bash
mkdir -p ~/projects
cd ~/projects
git clone https://github.com/alicamdal/vscode-hpc-tutorial.git
cd vscode-hpc-tutorial
```

Now open the folder in VS Code: *File > Open Folder...*, choose
`/home/s1234567/projects/vscode-hpc-tutorial`, click **OK**.

**Check:** the Explorer on the left shows the folders `demo` and `hpc`.

### C2. Create a virtual environment

A **virtual environment** is a private folder of Python packages for one project. Packages installed
here do not change other projects or the system Python. We call the folder `.venv`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

**Check:** the prompt now starts with `(.venv)`. Run `which python`; the path ends with
`vscode-hpc-tutorial/.venv/bin/python`.

> Each time you open a new terminal, activate the environment again with `source .venv/bin/activate`.

### C3. Install PyTorch

```bash
pip install torch numpy
```

This downloads about 3 GB and takes **about 3 minutes**. Wait until the prompt returns.

**Check:**

```bash
python -c "import torch; print(torch.__version__)"
```

prints a version such as `2.15.0+cu132`. The `cu132` part means "built for NVIDIA GPUs (CUDA 13.2)".

### C4. Try the GPU check on the login node

```bash
python hpc/check_gpu.py
```

The script says `GPU available  no` and `This is expected on the login node: it has no GPU.`
This is the point: **the login node has no GPU.** To use one, we ask Slurm for it.

### C5. Submit the job

Open `hpc/check_gpu.sbatch` in VS Code and read it. The `#SBATCH` lines ask Slurm for resources:

| Line | Meaning |
|---|---|
| `--partition=students` | the group of GPU computers for courses |
| `--gres=gpu:1` | one GPU |
| `--cpus-per-task=2`, `--mem=4G` | two CPU cores and 4 GB of memory |
| `--time=00:05:00` | stop the job after at most 5 minutes |
| `--output=gpu-check-%j.out` | write everything the job prints to this file (`%j` = job id) |

Submit it **from the project folder**:

```bash
sbatch hpc/check_gpu.sbatch
```

Slurm answers `Submitted batch job 123456`. Watch your job:

```bash
squeue --me
```

`PD` means waiting, `R` means running. When your job is no longer listed, it is finished. With 30
students on a few GPUs, waiting a minute or two is normal.

### C6. Read the result

A file `gpu-check-123456.out` appears in the Explorer. Open it, or run:

```bash
cat gpu-check-*.out
```

You see the `nvidia-smi` table and then something like this (your GPU may differ):

```
Where am I?
  machine                hpc-node08
  Slurm job id           123456
  partition              students
...
GPU
  GPU 0                  NVIDIA ...
    memory (VRAM)        ... GB total, ... GB free
...
Speed test: multiply two 4096 x 4096 matrices
  CPU                       ... ms
  GPU                       ... ms
  GPU is faster by       ... x
```

**Check:** the machine name is **not** `hpc-head1`. Your code ran on a compute node with a GPU.

### C7. Rules for the cluster

- No heavy work on the login node. Always use `sbatch` (or `srun` for short interactive sessions).
- Ask only for what you need (`--gres`, `--mem`, `--time`). Others are waiting.
- The cluster is not a backup. Copy results you want to keep to your laptop.
- Cluster documentation: <https://hpc.wiki.utwente.nl/eemcs-hpc>

When you are done, close the remote window: *File > Close Remote Connection*.

---

## Part D. Run the Vision Transformer on your laptop

Now we do the same steps (clone, environment, install, run) on your **own laptop**. The model is small
(about 23 MB) and runs on a normal CPU in about a second.

### D1. Clone the project on your laptop

Open a **new local** VS Code window (the bottom-left corner must **not** say SSH). Open a terminal:

```bash
cd ~/Documents
git clone https://github.com/alicamdal/vscode-hpc-tutorial.git
cd vscode-hpc-tutorial
```

Open the folder: *File > Open Folder...* and choose `Documents/vscode-hpc-tutorial`.

### D2. Create and activate a virtual environment

**Windows (PowerShell)**:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell says *running scripts is disabled on this system*, run this once and try again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

**Ubuntu**:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Then, on both systems:

```bash
python -m pip install --upgrade pip
```

**Check:** the prompt starts with `(.venv)`.

Tell VS Code to use this environment too: press `Ctrl+Shift+P`, type **Python: Select Interpreter**,
and choose the entry with `.venv`.

### D3. Install packages one by one

First PyTorch. Laptops usually have no NVIDIA GPU, so we install the smaller CPU version.

**Windows**:

```powershell
pip install torch
```

**Ubuntu** (without the extra address, pip downloads the large GPU version):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Then the Hugging Face library that downloads and runs the model:

```bash
pip install transformers
```

**Check:** `pip list` shows `torch` and `transformers`.

### D4. Install the rest from `requirements.txt`

Installing packages one by one is fine for two packages. Real projects need many, with known
versions. They are listed in a file called `requirements.txt`. Open it in VS Code and look at it.

```bash
pip install -r requirements.txt
```

pip says `Requirement already satisfied` for the packages you already have and installs the
rest (for example `matplotlib`).

> **Why did we install torch first?** On Ubuntu, `pip install -r requirements.txt` alone would pick
> the large GPU version of torch. Installing the right version first avoids that.

### D5. Run the demo

```bash
python demo/vit_demo.py
```

The first run downloads the model (about 23 MB). The program prints each step:

```
[Step 4] Cut the image into patches
  224 / 16 = 14 patches per side, 14 x 14 = 196 patches
  each patch has 16 x 16 x 3 = 768 pixel values

[Step 5] Run the transformer
  tokens per image: 197 = 1 class token + 196 patch tokens
  each token is a vector of 192 numbers (the embedding)
...
[Step 7] Prediction
   11.0 %  Egyptian cat
    7.8 %  tabby
...
```

**Check:** the folder `outputs/cat/` contains five images. Click them in the VS Code Explorer.

### D6. What the five figures show

| File | What you see | What to notice |
|---|---|---|
| `1_patches.png` | the 224 x 224 input with a 14 x 14 grid, and the patches pulled apart | a transformer reads an image as a **sequence** of 196 patches, like words in a sentence |
| `2_patch_embedding.png` | one patch and a bar chart of 192 numbers | the model turns each 768-number patch into a 192-number vector (the **embedding**) |
| `3_embeddings_pca.png` | all patch vectors as colours, before layer 1 and after layer 12 | before layer 1 the colours mostly show **position**; after 12 layers they show **content** (cat, background) |
| `4_attention.png` | where the class token looks, per head and over all layers | attention is how tokens share information; the decision draws on the object |
| `5_prediction.png` | the five most likely classes | the model knows the 1000 ImageNet classes |

### D7. Try it yourself

```bash
python demo/vit_demo.py --image demo/images/fox.jpg
python demo/vit_demo.py --image demo/images/bus.jpg
python demo/vit_demo.py --image demo/images/cat.jpg --patch 0
```

- The fox picture is a 19th-century **painting**, not a photo. Is the prediction still right?
- Take a photo with your phone and run the script on it. Which objects work well, and which do not?
- Look at `--patch 0` (top-left corner). How does its embedding differ from a patch on the cat?

---

## Part E. (Optional) An AI coding assistant

**GitHub Copilot** suggests code and answers questions about your code inside VS Code.

- Everyone with a GitHub account can use the limited **Copilot Free** plan.
- Verified students can get the **Copilot Student** plan for free through
  [GitHub Education](https://github.com/education). Verification can take a few days.
  See the [official guide](https://docs.github.com/en/copilot/how-tos/copilot-on-github/set-up-copilot/enable-copilot/set-up-for-students).

To start:

1. In VS Code, click the **Copilot** icon in the top bar (or the Accounts icon, bottom left) and
   **sign in with GitHub**.
2. Open the **Chat** view (`Ctrl+Alt+I`).
3. Open `demo/vit_demo.py` and try:
   - *"Explain what the function `attention_rollout` does, in simple words."*
   - *"Add a `--top` option that sets how many predictions are printed."*

Always read and test what the assistant writes. It can be wrong. Follow the course rules on using AI tools.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `ssh: connect to host ... Connection timed out` | You are not on the UT network. Connect eduVPN (Institute Access, University of Twente). |
| `Permission denied (keyboard-interactive)` | Add `-o PreferredAuthentications=password` (see B1), or check the `config` file (B2). |
| `Permission denied (password)` | Wrong username or password. The username is lowercase, for example `s1234567`. |
| Windows saved the config file as `config.txt` | Rename it: `Rename-Item $HOME\.ssh\config.txt config` |
| VS Code hangs on "Setting up SSH host" | Close the window and connect again. If it keeps failing: `Ctrl+Shift+P` > **Remote-SSH: Kill VS Code Server on Host**. |
| `sbatch: error: invalid partition` or `Invalid account` | Your account is not yet enabled for the `students` partition. Tell the TA. |
| The job stays `PD` for a long time | Other jobs use the GPUs. Wait; do not submit the job many times. |
| `py` or `python` not found (Windows) | Python is not on PATH. Re-run the installer, choose *Modify*, and tick *Add Python to environment variables*. |
| `python3 -m venv` fails (Ubuntu) | `sudo apt install python3-venv` |
| Ubuntu 22.04: Python is 3.10, torch will not install | Install Python 3.11 or newer, or upgrade to Ubuntu 24.04. Ask the TA. |
| `pip install torch` is very slow or huge on Ubuntu | You forgot `--index-url https://download.pytorch.org/whl/cpu` (see D3). |
| `ModuleNotFoundError: No module named 'torch'` | The environment is not active. Run the activate command (D2) and look for `(.venv)`. |
| The model download fails | Check your internet connection and run the script again. |
