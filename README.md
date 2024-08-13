## Using 3 workers, 
| GPU | SM | Mem | PID | C | Command | Memory |
|-----|----|----|-----|---|---------|--------|
| 0 | N/A | N/A | 4139816 | C | ...conda3/envs/removebg_dep/bin/python | 4788MiB |
| 0 | N/A | N/A | 4139954 | C | ...conda3/envs/removebg_dep/bin/python | 4788MiB |
| 0 | N/A | N/A | 4141717 | C | ...conda3/envs/removebg_dep/bin/python | 4788MiB |

run on local: sh start_server_local.sh

testing the server: sh test_one.sh

Stress Testing local: cd tests && locust -f locustfile.py --host=https://internal-ems.echo.tech/removebg
Stress Testing remote: cd tests && locust -f locustfile.py --host=http://192.168.12.68:8080

curl http://127.0.0.1:8080/status