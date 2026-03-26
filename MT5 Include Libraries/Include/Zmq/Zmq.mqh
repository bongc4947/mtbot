//+------------------------------------------------------------------+
//| Module: Zmq.mqh                                                  |
//| Patched for newer MT5 uchar[] UTF-8 handling                     |
//+------------------------------------------------------------------+
#property strict

#include <Mql/Lang/Native.mqh>
#include "AtomicCounter.mqh"
#include "Z85.mqh"
#include "Socket.mqh"

#define ZMQ_VERSION_MAJOR 4
#define ZMQ_VERSION_MINOR 2
#define ZMQ_VERSION_PATCH 0

#define ZMQ_HAS_CAPABILITIES 1

#import "libzmq.dll"
int zmq_errno(void);
intptr_t zmq_strerror(int errnum);
void zmq_version(int &major,int &minor,int &patch);
int zmq_has(const uchar &capability[]);
#import

class Zmq
  {
protected:
   static bool       has(string cap);
public:
   static bool       hasIpc() {return has("ipc");}
   static bool       hasPgm() {return has("pgm");}
   static bool       hasTipc() {return has("tipc");}
   static bool       hasNorm() {return has("norm");}
   static bool       hasCurve() {return has("curve");}
   static bool       hasGssApi() {return has("gssapi");}

   static int        errorNumber() {return zmq_errno();}
   static string     errorMessage(int error=0);
   static string     getVersion();
  };

bool Zmq::has(string cap)
  {
   uchar capstr[];
   StringToUtf8(cap,capstr);
   bool res=(ZMQ_HAS_CAPABILITIES==zmq_has(capstr));
   ArrayFree(capstr);
   return res;
  }

string Zmq::errorMessage(int error)
  {
   intptr_t ref=error>0?zmq_strerror(error):zmq_strerror(zmq_errno());
   return StringFromUtf8Pointer(ref);
  }

string Zmq::getVersion(void)
  {
   int major,minor,patch;
   zmq_version(major,minor,patch);
   return StringFormat("%d.%d.%d", major, minor, patch);
  }
