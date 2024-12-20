For Windows
In Container:
```
sudo passwd vscode
sudo xrdp-sesman
sudo xrdp --no-daemon
```

In vscode: Forward port 3389 (defaults to loaclhost:3390) 

On Windows machine: Connect rdp session to localhost:3390
In remote desktop login: vscode/password set above

In Container terminal: xauth list
ex: 00e7a04c797f/unix:10

In container terminal python debug console 
```
export DISPLAY=:10
```


For Linux:
On host:  xauth list
In container: xauth add :0 MIT-MAGIC-COCKIE-1 xxxxxxx

