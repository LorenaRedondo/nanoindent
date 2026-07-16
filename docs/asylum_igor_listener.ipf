#pragma TextEncoding = "UTF-8"
#pragma rtGlobals=3

// ============================================================================
// nanoindent — reference Igor listener for the Asylum adapter
// ============================================================================
//
// Drop this procedure into your Asylum (MFP-3D / Cypher) Igor experiment to let
// nanoindent's AsylumAdapter drive the microscope over TCP. It implements the
// command grammar spoken by nanoindent.adapters.SocketIgorBridge:
//
//     EXEC <igor command>   -> OK | ERR <msg>       (runs Execute/Q "<command>")
//     CAPS                  -> CAPS scan_size_nm=.. points=.. lines=.. scan_rate_hz=.. mode=..
//     FRAME                 -> IMG <h> <w> <itemsize>\n<h*w*8 little-endian float64 bytes>
//                              | NONE
//     FORCE <dy> <dx>       -> FD <n>\n<2n little-endian float64: z_nm,F_pN interleaved>
//     CLOSE                 -> (closes the connection)
//
// TRANSPORT
//   Igor has no built-in TCP server, so wire the four Socket* helpers below to a
//   TCP XOP available on your system (e.g. a community TCP/networking XOP). If no
//   XOP is available, use the FILE-POLLING variant described at the bottom — the
//   command grammar is identical; only the transport changes, and it needs no XOP.
//
// !!  VERIFY  !!
//   The Asylum function / global-parameter names used below (SetScanOffset,
//   PV("SetpointDefl", ...), Approach, Withdraw, StartFrame/StopFrame, the scan
//   master variables, and the force-curve trigger) are PLACEHOLDERS that match
//   nanoindent's adapter. Replace them with the exact names from the Asylum
//   Programmer's Guide for your software version. The nanoindent side is fully
//   tested against a mock listener; only these names need substituting.
// ============================================================================

// ---- helpers you must implement against your TCP XOP -----------------------
// Each returns/accepts plain strings; binary payloads are written raw.
Function/S Socket_ReadLine(sock)          // read one '\n'-terminated line
    Variable sock
    // TODO: return the next line from the socket (without the trailing newline)
    return ""
End

Function Socket_WriteStr(sock, s)         // write a string (append '\n' yourself)
    Variable sock
    String s
    // TODO: send s over the socket
End

Function Socket_WriteBinary(sock, w)      // write a numeric wave as raw float64 LE
    Variable sock
    WAVE w
    // TODO: send w's bytes (double precision, little-endian) over the socket
End

// ---- capability report -----------------------------------------------------
Function/S niCapsString()
    // VERIFY: pull these from the Asylum scan master variables.
    NVAR/Z scanSize = root:packages:MFP3D:Main:Variables:ScanSize      // metres
    NVAR/Z points   = root:packages:MFP3D:Main:Variables:PointsLines
    NVAR/Z scanRate = root:packages:MFP3D:Main:Variables:ScanRate      // Hz (lines/s)
    Variable sizeNm = (NVAR_Exists(scanSize) ? scanSize*1e9 : 0)
    Variable pts    = (NVAR_Exists(points)   ? points       : 0)
    Variable rate   = (NVAR_Exists(scanRate) ? scanRate     : 0)
    String mode = "AC"                                                 // VERIFY
    return "CAPS scan_size_nm=" + num2str(sizeNm) + " points=" + num2str(pts) \
         + " lines=" + num2str(pts) + " scan_rate_hz=" + num2str(rate) \
         + " mode=" + mode
End

// ---- most-recent scan frame as a 2-D wave ----------------------------------
Function/WAVE niCurrentFrame()
    // VERIFY: return the latest captured height frame (a 2-D wave). Asylum stores
    // captured images under root:... — point this at your height channel.
    WAVE/Z frame = root:FrameImage    // placeholder
    return frame
End

// ---- run a single force curve at an offset, return z(nm) & F(pN) ------------
Function niForceCurve(dyNm, dxNm, zOut, fOut)
    Variable dyNm, dxNm
    WAVE zOut, fOut
    // VERIFY: move to the point, trigger a single force curve, then copy the
    // captured Z-sensor (-> nm) and deflection (-> force in pN) into zOut/fOut.
    Execute/Q "DoSingleForce()"        // placeholder trigger
End

// ============================================================================
// main dispatch loop — call niServe(sock) after your XOP accepts a connection
// ============================================================================
Function niServe(sock)
    Variable sock
    String line, verb, rest, reply
    Variable dy, dx, hh, ww
    do
        line = Socket_ReadLine(sock)
        if (strlen(line) == 0)
            break
        endif
        verb = StringFromList(0, line, " ")
        rest = line[strlen(verb)+1, inf]

        strswitch (verb)
            case "CLOSE":
                return 0
            case "EXEC":
                Execute/Q/Z rest
                Socket_WriteStr(sock, V_flag == 0 ? "OK" : "ERR igor " + num2str(V_flag))
                break
            case "CAPS":
                Socket_WriteStr(sock, niCapsString())
                break
            case "FRAME":
                WAVE/Z fr = niCurrentFrame()
                if (!WaveExists(fr))
                    Socket_WriteStr(sock, "NONE")
                else
                    hh = DimSize(fr, 0); ww = DimSize(fr, 1)
                    Socket_WriteStr(sock, "IMG " + num2str(hh) + " " + num2str(ww) + " 8")
                    Socket_WriteBinary(sock, fr)   // as float64 little-endian
                endif
                break
            case "FORCE":
                dy = str2num(StringFromList(0, rest, " "))
                dx = str2num(StringFromList(1, rest, " "))
                Make/O/N=0/D niZ, niF
                niForceCurve(dy, dx, niZ, niF)
                // interleave z,F into one wave and stream it
                Variable n = numpnts(niZ)
                Make/O/N=(2*n)/D niFD
                niFD[0,;2] = niZ[p/2]
                niFD[1,;2] = niF[(p-1)/2]
                Socket_WriteStr(sock, "FD " + num2str(n))
                Socket_WriteBinary(sock, niFD)
                break
            default:
                Socket_WriteStr(sock, "ERR unknown " + verb)
        endswitch
    while (1)
    return 0
End

// ============================================================================
// FILE-POLLING VARIANT (no XOP required)
// ----------------------------------------------------------------------------
// Instead of a socket, use nanoindent's file bridge (a directory both sides can
// see):
//   * the adapter writes a one-line command file  cmd_<n>.txt  ("EXEC ...", etc.);
//   * a background task (CtrlNamedBackground) polls the directory, runs the same
//     dispatch as niServe but reading the line from the file, and writes the
//     reply to  reply_<n>.txt  (and frame/force data to  data_<n>.bin).
// The command grammar and the VERIFY notes above are unchanged. Register with:
//   CtrlNamedBackground niPoll, period=6, proc=niPollTask   // ~10 Hz
// ============================================================================
