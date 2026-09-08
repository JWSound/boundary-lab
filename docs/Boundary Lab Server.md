# Boundary Lab Server

Run solves on another workstation or hosted machine while working in the Boundary
Lab application. The server chooses its solver automatically; clients do not need
to know its hardware. This feature is currently a preview. Polar and balloon
results are supported; observation planes are not yet available for remote solves.

## Connect from the application

1. Open **Preferences → Application** and set **Solver** to **Boundary Lab Server**.
2. Enter the server **Address**, such as `http://192.168.1.20:8765` on a private LAN
   or your hosting provider's HTTPS service address.
3. Enter the **Access key** if required. Leave it empty for a private server
   configured without a key.
4. Click **Check connection**. If the server is starting, wait and check again.
5. Save Preferences and use the usual solve and stop controls. Results appear in
   the existing plots.

**Generate** creates a key, **Copy** copies it, and **Show** reveals the masked
value. Generating a key does not configure the server automatically; set the same
key on the server as described below.

Save the key locally in a safe place so you can paste it again next time. Boundary
Lab remembers the address but keeps the key only for the current application
session. It does not save the key in preferences or the OS credential store.

To resume local solving, choose a local solver in Preferences. The server address
remains saved for later use.

## Prepare the server machine

Install Boundary Lab and Julia on the machine that will run the solves, following
[Installation and Setup](Installation%20and%20Setup.md). From the Boundary Lab
Python environment, prepare the CPU runtime:

```bash
python -m beat_engine instantiate --backend cpu
```

For a GPU server, also prepare its matching runtime:

```bash
python -m beat_engine instantiate --backend cuda
# Or, for a supported AMD GPU:
python -m beat_engine instantiate --backend rocm
```

Install the compatible driver/SDK on the server. See [CUDA setup](advanced/beat-engine-CUDA.md)
or [ROCm setup](advanced/beat-engine-rocm.md) for requirements. Clients do not need
the server's GPU runtime. Keep client and server updated together, and restart the
server after installing or repairing a runtime.

## Private LAN or VPN

On the server machine, run:

```bash
python -m blab.cli server --root runs/server-jobs --mode private-network --host 0.0.0.0 --port 8765
```

Keep this process running while clients use it. The `--root` folder stores server
jobs; choose a location with enough free space. `--host 0.0.0.0` accepts connections
through the machine's network interfaces. Clients use the machine's actual LAN/VPN
address, such as `http://192.168.1.20:8765`, not `0.0.0.0`. Allow the chosen port
through the server firewall for your intended clients.

An access key is optional. Without one, anyone who can reach the service can
submit, read, and cancel jobs. HTTP does not encrypt the key or model data; use it
only on a trusted private network or through a VPN/SSH tunnel.

To require a key, generate and copy one in Boundary Lab, then set
`BLAB_SERVER_TOKEN` in the server environment before startup. For example, in
PowerShell, replace the placeholder with your copied key:

```powershell
$env:BLAB_SERVER_TOKEN = 'paste-your-generated-key-here'
```

Start the server from that terminal and enter the matching key in each client.
For same-machine testing, omit `--host 0.0.0.0` and connect to
`http://127.0.0.1:8765`.

The default server policy chooses an available GPU, then CPU. Operators can add
`--backend cpu`, `--backend cuda`, or `--backend rocm` to pin the BEM solver. Pure
interior FEM uses CPU. Clients still only choose the server address.

## Hosted server or container

For locally built CPU/CUDA images, see [Docker setup](Docker.md).

For locally built CPU/CUDA images, see [Docker setup](Docker.md).

Use these settings in an environment with Boundary Lab, Julia, and the required
solver runtime already installed:

1. In Boundary Lab Preferences, click **Generate**, then **Copy**, and save the key locally.
2. Paste it into the container's `BLAB_SERVER_TOKEN` secret.
3. Start the server in hosted mode:

   ```bash
   python -m blab.cli server --root /data/jobs --mode hosted --host 0.0.0.0 --port 8765
   ```

4. Configure the provider's HTTPS proxy to forward to container HTTP port `8765`.
   Paste the provider's HTTPS service address into Boundary Lab.
5. Enter the same key and click **Check connection**.

Hosted mode requires a key. The provider manages HTTPS; Boundary Lab has no
certificate files to configure. Keep the container's plain HTTP port behind that
protected connection instead of also exposing it directly to the Internet. Use
persistent storage for `/data/jobs` if jobs should survive container replacement.

SSH and VPN are alternatives: connect to the forwarded localhost HTTP address for
SSH, or the server's private HTTP address for a VPN. Keep the hosted key configured
in both cases. All clients with the key share access to server jobs.

To replace a key, update the server secret, restart the server, and paste the new
key into clients.

## Everyday operation and troubleshooting

The server currently runs one job at a time. If it is busy, wait for that job to
finish and submit again; additional jobs are not queued. Closing a client does not
necessarily stop a submitted job, so use **Stop** when you intend to cancel it.

Server job files remain until you remove them. Stop the server before removing
unneeded job directories. Use a separate job root for each server process.

| Problem | What to check |
| --- | --- |
| Cannot connect | Confirm the server is running, the address and port are correct, and the firewall or provider proxy permits the connection. |
| Server is starting | Allow startup to finish, then check again. |
| Authentication required | Paste the same key configured in the server's `BLAB_SERVER_TOKEN`. The app does not remember it after exit. |
| Server is not ready to solve | Check the server console, prepare or repair its BEAT runtime, and restart the server. |
| HTTPS certificate error | Use the provider's correct HTTPS service address. Boundary Lab does not bypass certificate verification. |
| Incompatible server | Update both application and server to compatible versions. |

For command-line submission, see the [headless project workflow](advanced/cli-workflow.md#headless-project-workflow).
For protocol details, job behavior, and testing, see the
[server developer reference](advanced/boundary-lab-server.md).
