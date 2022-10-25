FROM proteusdb/pelago-build:cuda11.3-llvm14.0.0

RUN apt update  -y && \
    apt install -y gosu vim nano rsync tar gdb && \
    apt autoremove -y && \
    apt autoclean && \
    rm -rf /var/lib/apt/lists/* /var/log/dpkg.log

RUN ( \
    echo 'LogLevel DEBUG2'; \
    echo 'PermitRootLogin yes'; \
    echo 'PasswordAuthentication yes'; \
    echo 'Subsystem sftp /usr/lib/openssh/sftp-server'; \
  ) > /etc/ssh/sshd_config_proteus \
  && mkdir /run/sshd


# add user for ssh access
RUN useradd -m user && yes password | passwd user

CMD ["/usr/sbin/sshd", "-D", "-e", "-f", "/etc/ssh/sshd_config_proteus"]