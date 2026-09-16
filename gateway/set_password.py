"""Write the browser-login password hash (scrypt) to argv[1], mode 0600. Used by `llm passwd`."""
import getpass, hashlib, os, secrets, sys

pw = os.environ.get("LLM_PASSWORD") or getpass.getpass("new browser password: ")
if len(pw) < 8:
    sys.exit("llm: password must be at least 8 characters")
salt = secrets.token_bytes(16)
h = hashlib.scrypt(pw.encode(), salt=salt, n=2**14, r=8, p=1, dklen=32).hex()
fd = os.open(sys.argv[1], os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
os.write(fd, f"{salt.hex()} {h}\n".encode())
os.close(fd)
print(f"llm: wrote {sys.argv[1]} (scrypt hash, mode 0600)")
