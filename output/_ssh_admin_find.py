import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.0.153.222", username="peter", password="iamroot6025", timeout=15, allow_agent=False, look_for_keys=False)
cmds = [
    "grep -r minio ~/ 2>/dev/null | head -20",
    "find /home/peter -name '*minio*' 2>/dev/null | head -20",
    "ls /home/peter/ 2>/dev/null",
]
for cmd in cmds:
    print("===", cmd, "===")
    _, stdout, stderr = c.exec_command(cmd, timeout=30)
    print(stdout.read().decode()[:3000])
    print(stderr.read().decode()[:500])
c.close()
