#!/usr/bin/env python

import jpype
import jaydebeapi
import os
import warnings

commithash = os.environ['CI_COMMIT_SHA']
mysecret = os.environ['BENCHDBPASS']
avatica = os.environ['JAVA_AVATICA_CLASSPATH']

if "classpath" in locals() or "classpath" in globals():
    classpath = classpath + ":" + str(avatica)
else:
    classpath = str(avatica)

classpath = classpath + ':' + '/usr/share/java/postgresql.jar'

if not jpype.isJVMStarted():
    assert(avatica in classpath)
    jpype.startJVM(jpype.getDefaultJVMPath(), "-Djava.class.path=" + classpath)
else:
    warnings.warn("JVM already started. Keeping old JVM configuration.")


conn = jaydebeapi.connect('org.postgresql.Driver', 'jdbc:postgresql://db:5432/postgres', ['runner', mysecret])
curs = conn.cursor()

class NULL:
    def __str__(self):
        return "NULL"
    __repr__ = __str__

def escape(s):
    for c in "\"'":
        s = s.replace(c, "_")
    return s

def escape2(s):
    if s is None:
        return NULL()
    else:
        return str(escape(s))

curs.execute("INSERT INTO results_repo(commithash, repourl, branch) VALUES" +str((commithash, os.environ['CI_PROJECT_URL'], escape(os.environ['CI_COMMIT_REF_NAME']))))

class DEFAULT:
    def __str__(self):
        return "DEFAULT"

    __repr__ = __str__

host='http://localhost:8081'
proteus = jaydebeapi.connect(jclassname="org.apache.calcite.avatica.remote.Driver",
                   url="jdbc:avatica:remote:url=" + host + ";serialization=PROTOBUF",
                   driver_args={})

cursprot = proteus.cursor()
cursprot.execute('SELECT * FROM SessionTimings')
for t in cursprot.fetchall():
    t = tuple(int(x) if isinstance(x, (int, long)) else escape2(x) for x in t)
    try:
        curs.execute('INSERT INTO SessionTimings VALUES ' + str((os.environ['CI_PROJECT_URL'], commithash) + t))
    except:
        print('INSERT INTO SessionTimings VALUES ' + str((os.environ['CI_PROJECT_URL'], commithash) + t))
        raise
