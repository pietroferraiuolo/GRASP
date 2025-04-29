!c    Questo è un esempio di programma completo (King isotropo)
!c    Leggilo bene e poi fallo girare e confronta i risultati in letteratura.
!c    Infine trasforma il programma per la funzione di Boltzmann anisotropa
!c    e confronta i risultati con l'articolo

!C    CALCOLA COL FORMALISMO NEWTONIANO LE CONFIGURAZIONI DI EQUILIBRIO
!C    GRAVITAZIONALE DI UN GAS. LA FUNZIONE DI DISTRIBUZIONE DI KING E'
!C    TRATTATA COME NELL'ARTICOLO DEL 1966
!C    KING I.P. 1966 AJ 71 64

      PROGRAM MAIN
        IMPLICIT REAL*8(A-H,O-Z)
        REAL*4 temp1,temp2
        COMMON/PARFCN/WWWW
        COMMON/ESTREMI/ESTR1,ESTR2
        COMMON/VALINI/W0,RHO0
        COMMON/GAUSS/INDGAU
        COMMON/COSTANTI/PAI
        COMMON/RESULTS/X(8000),W(8000),DW(8000),RAPP(8000),NFN
        COMMON/VEL/v2(8000)
        DIMENSION Y(2),DERY(2),PRMT(5),AUX(16,2)
        !      DIMENSION PARW0(2000) !221
        ! subroutines do not need external declaration!!!!
        EXTERNAL FCT,OUTP, INTGAU!, SURFDENS
        external :: fcnqd, fcnqd1, fcnqd2
        external :: fu1, fu2, fu3, fu4, funzerr, funzione
        real :: w0
        character(len=10) :: arg
       ! external :: FU1,FU2,FU3,FU4,FCNQD1,funzerr, funzione
        COMMON/SURF/t(8000),phi(8000),sigma(8000),psi(8000),S_S0(8000)
        COMMON/SURF2/csi(8000)
        COMMON/CV/Cv0(8000),Cv(8000),Cvqt(8000),Ctot0,Ctot1,Ctot2
        COMMON/CV2/xMhat,xnhat(8000),x0Cv(10),xi0Cv(10),potch(8000)
        COMMON/PARFU2/qq
        DIMENSION xN(8000)
        DIMENSION phi_g(8000)
        DIMENSION phi_g2(8000),eg2(8000),xmu_r(8000),secpezz(8000)
        COMMON/ENERGY/xk(8000),eg(8000),ephi(8000),xkt,egt,ephit,Etot,Vir2,et_r(8000),Etot1
        COMMON/TEST/C,D
        COMMON/CC/yEt_Mv02(10000),xs2_v02(10000),fw0,xsave,ysave,w0save
        COMMON/Mrho0hat/rho0hat,xMcap,xmu,xnhat0,conc
        COMMON/SURFEKIN/uk(8000),sigekin(8000),Sk_Sk0(8000)
        DIMENSION parw0(6)!135)
        CHARACTER(LEN=10), DIMENSION(6) :: NAMESID !dimension must be 135

!     Analyze only a subset of known GCs
   DATA PARW0/6.18d0,6.22d0,7.41d0,6.48d0,&
   2.09d0,6.29d0/

   DATA NAMESID/"NGC 4372","NGC 5139","NGC 6121","NGC 6656",&
   "Pal 5","GLIMPSE02"/

   OPEN(UNIT=1,FILE='params.dat',STATUS='replace') !,ACCESS='append')
   CLOSE(UNIT=1,STATUS='keep')
   OPEN(UNIT=1,FILE='profiles.dat',STATUS='replace')!,ACCESS='append')
   CLOSE(UNIT=1,STATUS='keep')
   OPEN(UNIT=1,FILE='CvNtK.dat',STATUS='replace')
   CLOSE(UNIT=1,STATUS='keep')
   OPEN(UNIT=1,FILE='Cv.dat',STATUS='replace')
   CLOSE(UNIT=1,STATUS='keep')
   OPEN(UNIT=1,FILE='phi.dat',STATUS='replace')
   CLOSE(UNIT=1,STATUS='keep')
   OPEN(UNIT=1,FILE='Er.dat',STATUS='replace')
   CLOSE(UNIT=1,STATUS='keep')
   OPEN(UNIT=1,FILE='Etot.dat',STATUS='replace')
   WRITE(1,303)'w0 K Egr Eeff Etot s2_v02 Et_Mv02 Mcap Vir conc Etot1'
   CLOSE(UNIT=1,STATUS='keep')
   OPEN(UNIT=1,FILE='CalCurve.dat',STATUS='replace')
   CLOSE(UNIT=1,STATUS='keep')
   OPEN(UNIT=1,FILE='x0Cv.dat',STATUS='replace')
   WRITE(1,303)'w0 x1 x2 x3 xi1 xi2 xi3 conc c'
   CLOSE(UNIT=1,STATUS='keep')
   OPEN(UNIT=1,FILE='Skin.dat',STATUS='replace')
   CLOSE(UNIT=1,STATUS='keep')

   temp1=0.0d0
   temp2=0.0d0
   call second(temp1)

   PRINT*, 'Executing Single-Mass King model integration routine.'
   IW1=1
   IW2=6!135!160    !221
   IWSTEP=1

!    l'indice che precisa che subroutine per gli integrali si vuole usare
!    il valore 3 corrisponde a 80 punti

   indgau = 4

!    l'errore relativo si decide a priori e ci da la precisione voluta
   relerr = 1.d-9


   estr1  = 0.d0
   pai    = 3.141592654d0

      ! Leggi l'argomento dalla riga di comando
    call get_command_argument(1, arg)
    read(arg, *) w0

    ! Stampa il valore letto
    print *, 'W0 =', w0

      w(1) = w0
      dw(1) = 0.d0
      wwww = w0
      estr2= w0
      call intgau(fcnqd,qd)
      rho0 = qd
         rapp(1) = qd/rho0

!c ********************** INIZIALIZZAZIONE **************************
!c             (per evitare la singolarità iniziale)
      AAA  = -1.5d0
      XMIN = (-relerr*w0/AAA)**0.5d0
      IIII = DLOG10(XMIN)
      if(IIII.lt.0) iiii = IIII-1
      XMIN = 10.d0**IIII

      X(2) = XMIN
      w(2) = w0+AAA*XMIN**2.d0
      dw(2)= 2.D0*AAA*XMIN

      wwww = w(2)
      estr2= w(2)

      call intgau(fcnqd,qd)
      rapp(2) = qd/rho0

!c ******************* FINE INIZIALIZZAZIONE ************************
!********************************************************************
      nfn   = 2
      passo = x(nfn)/10.0d0!10!2
      xmax  = x(nfn)*10.0d0!10!2

 1111    CONTINUE

      ABSERR  = relerr * w(nfn)
      if(abserr.lt.1.d-12) abserr=1.d-12

      PRMT(1) = X(NFN)
      PRMT(2) = XMAX
      PRMT(3) = PASSO
      PRMT(4) = ABSERR

      y(1)     = w(nfn)
      y(2)     = dw(nfn)
      dery(1)  = 0.5d0
      dery(2)  = 0.5d0

      CALL DHPCG(PRMT,Y,DERY,2,IHLF,FCT,OUTP,AUX,FCNQD)

      if(ihlf.lt.11) goto 1122

      print*,'Bisections > 10',nfn,w0!,x(nfn)
      passo = passo/10.d0!10!200
      xmax  = x(nfn)+(xmax-x(nfn))/10.d0!10!200
      goto 1111

 1122   continue

      if(nfn.eq.8000) goto 3333
      if(w(nfn).lt.0.d0)   GO TO 1234

      passo = passo*10.d0!10!2
      xmax  = xmax *10.d0!10!2

      goto 1111

 1234   continue
      xtest = (dw(nfn-1)*x(nfn-1)-w(nfn-1))/dw(nfn-1)

   !    controllo precisione sulla routine di uscita per determinare il raggio

      if(dabs((xtest-x(nfn))/xtest).lt.1.d-9) goto 2222

      xmax  =  x(nfn)
      passo = (x(nfn)-x(nfn-1))/10.d0!10!200
      nfn = nfn - 1
      goto 1111

 2222    continue

   !        print*,nfn
      ! Calculate concentration in king units /r_t/r_k and scale positions to than value to have xi = r/r_t
      conc  =  (xtest+x(nfn))/2.d0
      DO kk=1, nfn
         csi(kk)=x(kk)/conc
         IF(kk==nfn) csi(kk)=conc/conc
      END DO

      xlogc =   dlog10(conc)
      xmu1  = -(4.d0*pai/9.d0)*(x(nfn-1)**2.d0)*dw(nfn-1)
      xmu2  = -(4.d0*pai/9.d0)*(x(nfn)**2.d0)*dw(nfn)
      xmu   =  (xmu1+xmu2)/2.d0
      C1 =-(x(nfn-1)*dw(nfn-1))
      C2= -(x(nfn)*dw(nfn))
      C = (C1+C2)/2.0d0         !-phi_R/sigma1^2
      D = C + w0                   !-phi_0/sigma1^2

   !  Evalute projected surface density Sigma(R)/Sigma(0)
      call surfdens(fcnqd)

   !  Evaluate mean square velocity <v^2>/<v^2>_w->inf=<v^2>/3sigma^2
      call v2mean(fcnqd,fcnqd2)

   !  Evaluate heat capacity profile and total value
      call calorespecifico(fcnqd1,fu1,fu2,fu3,fu4)!,funzerr)

      xKb=1.0d0
      Ct0_NtK=xKb*Ctot0
      Ct1_NtK=xKb*Ctot1!/xmu
      Ct2_NtK=xKb*Ctot2       !/xmu

   !  Evaluate energies
      call energie(fu1,fu3,fcnqd1)

   !        PRINT*,Ct0_NtK,Ct1_Ntk,Ct2_Ntk!,Ctot1,Ctot2

      IF(w0.eq.8.20d0)THEN
   !           xm_k= 0.5295d0       !w0=8.0
         xm_k=0.54274d0
         DO k=1,nfn
            phi_g(k)=(-C-w(k))!/xm_k !=phi(r)/k\theta
   !              PRINT*,xm_k*phi_g(k)+potch(k),potch(k),xm_k*phi_g(k)
         ENDDO
      ENDIF

   !     Uso metodi diversi per calcolare Egr
   !     3: confronto il profilo di phi_gr(r) ottenuto prima con quello generico
      xmu_r(1)=0.d0

      DO l=1,nfn-1
         b1=0.d0
         h1=0.d0
         sum1=0.d0
         IF(l.gt.1)THEN
            DO i=2,l
               b1=rapp(i-1)*x(i-1)*x(i-1)+rapp(i)*x(i)*x(i)
               h1=x(i)-x(i-1)
               sum1=sum1+b1*h1*0.5d0
            ENDDO
         ENDIF
         xmu_r(l)=sum1
         b2=0.d0
         h2=0.d0
         sum2=0.d0
         DO m=l,nfn-1
            b2=rapp(m+1)*x(m+1)+rapp(m)*x(m)
            h2=x(m+1)-x(m)
            sum2=sum2+b2*h2*0.5d0
         ENDDO
         secpezz(l)=sum2

      phi_g2(l)=-(9.d0/xm_k)*((sum1/x(l))+sum2)
      ENDDO
      DO k=2,nfn-1
         eg2(k)=xnhat(k)*(-9.d0)*((xmu_r(k)/x(k)))!+secpezz(k))!*(-C-w(k))
      ENDDO
      eg2(1)=0.d0!0.5d0*xnhat(1)*(-9.d0)*(secpezz(1))
      eg2(nfn)=0.0d0
      b=0.0d0
      h=0.d0
      sum=0.0d0
      DO l=1,nfn-1
         b=eg2(l+1)*csi(l+1)*csi(l+1)+eg2(l)*csi(l)*csi(l)
         h=csi(l+1)-csi(l)
         sum=sum+b*h*0.5d0
      ENDDO
      egt2=sum*4.d0*pai
      Shat=IW*0.01d0
   !    scrive i parametri principali delle configurazioni in un unico file
   !    w0, concentrazione, log10 concentrazione, mu (massa adimensionale)
      OPEN(UNIT=1,FILE='params.dat',STATUS='old',ACCESS='append')
      WRITE(1,301) w0,conc,xlogc,xmu
      CLOSE(UNIT=1,STATUS='keep')
      OPEN(UNIT=1,FILE='profiles.dat',STATUS='old',ACCESS='append')
      WRITE(1,303)'x xi w rho_rho0 SD v2 logc=',xlogc
         DO jj=1,nfn
            WRITE(1,306) x(jj),csi(jj),w(jj),rapp(jj),s_s0(jj),v2(jj)
         ENDDO
      CLOSE(UNIT=1,STATUS='keep') !,dispose='keep')
   !      ENDIF
      IF((w0==0.8d0).OR.(w0==1.35d0).OR.(w0==2.0d0).OR.&
      (w0==3.5d0).OR.(w0==5.d0).OR.(w0==6.0d0).OR.&
      (w0==8.0d0).OR.(w0==8.2d0).OR.(w0==40.0d0))THEN

         OPEN(UNIT=1,FILE='Cv.dat',STATUS='old',ACCESS='append')
         WRITE(1,303)'x xi w  dw Cv0 Cv Cvqt w0=',w0
         DO m=1,nfn
            WRITE(1,302)x(m),csi(m),w(m),dw(m),Cv0(m)/xnhat(m),&
            Cv(m)/xnhat(m), Cvqt(m)/xnhat(m)
         ENDDO
         CLOSE(UNIT=1,STATUS='keep')
      ENDIF

      OPEN(UNIT=1,FILE='CvNtK.dat',status='old',access='append')
      WRITE(1,302) w0,Ct0_NtK,Ct1_NtK,Ct2_NtK,conc,x0Cv,xi0Cv
      CLOSE(UNIT=1,STATUS='keep')

      OPEN(UNIT=1,FILE='x0Cv.dat',status='old',access='append')

      WRITE(1,302) w0,x0Cv(1),x0Cv(2),x0Cv(3),xi0Cv(1),xi0Cv(2),xi0Cv(3)&
      ,conc,xlogc
      CLOSE(UNIT=1,STATUS='keep')

      IF(w0==8.20d0)THEN
         OPEN(UNIT=1,FILE='phi.dat',status='old',access='append')
         DO k=1,nfn
            WRITE(1,302)x(k),csi(k),w(k),dw(k),phi_g(k)
         ENDDO
         CLOSE(UNIT=1,STATUS='keep')
      ENDIF
      espo=4.d0/3.d0
      s2_v02=(1.d0/(xMcap**espo))
      Et_Mv02=s2_v02*(Etot1/xMcap)
      xkt=s2_v02*(xkt/xMcap)
      egt=s2_v02*(egt/xMcap)
      ephit=s2_v02*(ephit/xMcap)

      OPEN(UNIT=1,FILE='Er.dat',status='old',access='append')
      OPEN(UNIT=2,FILE='Etot.dat',status='old',access='append')
      IF((w0==0.8d0).OR.(w0==1.35d0).OR.(w0==2.0d0).OR.&
      (w0==3.5d0).OR.(w0==5.d0).OR.(w0==6.0d0).OR.&
      (w0==8.0d0).OR.(w0==8.2d0).OR.(w0==40.0d0))THEN
         DO k=1,nfn
            WRITE(1,302)x(k),csi(k),w(k),xk(k),eg(k),ephi(k),et_r(k)
         ENDDO
      ENDIF

      WRITE(2,302)w0,xkt,egt,ephit,Etot,s2_v02,Et_Mv02,xMcap,& !hat,!(shat*Etot/(4.d0*pai*xMhat))
      Vir2,conc, Etot1

      CLOSE(UNIT=1,STATUS='keep')
      CLOSE(UNIT=2,STATUS='keep')

   !  Evaluate projected kinetic energy density
      call surf_dens_ekin(fu1)

      OPEN(UNIT=1,FILE='Skin.dat',status='old',access='append')
      WRITE(1,305)'NameID=',NAMESID(iw)
      WRITE(1,303)'w0=',w0
      WRITE(1,303)'x xi w Sk sk2'
      DO jj=1,nfn
         WRITE(1,302) x(jj),csi(jj),w(jj),Sk_Sk0(jj),sigekin(jj)
      ENDDO
      CLOSE(UNIT=1,STATUS='keep') !,dispose='keep')

301   FORMAT(1pd9.2,1pd10.3,1x,1p2d15.8)
302   FORMAT(1p11d16.8)
303   FORMAT(A70,1p3d15.8)
304   FORMAT(1p5d16.8)
305   FORMAT(A30,A30)
306   FORMAT(1p6e15.8)

 3333   CONTINUE
      CALL SECOND(TEMP2)
      T_imp=0.0d0
      T_imp=(temp2-temp1)
      PRINT*,'Execution time(s)=',T_imp
      DO j=1,2
         !print *, char(7)
         call sleep(1)
      ENDDO
      STOP
   !ENDDO
END PROGRAM

   function funzione(y)
      implicit real*8 (A-H, O-Z)
    !  real*8 :: y
      funzione = y
      return
      end

   SUBROUTINE prova(funzione, testo, y)
      character(5) :: testo
      real*8 :: y, funzione
      !EXTERNAL funzione
      print*,testo
      print*,funzione(y)
      return
      end

   SUBROUTINE surf_dens_ekin(fcn)
   IMPLICIT REAL*8(A-H,O-Z)
   COMMON/RESULTS/X(8000),W(8000),DW(8000),RAPP(8000),NFN
   COMMON/PARFCN/WWWW
   COMMON/ESTREMI/ESTR1,ESTR2
   COMMON/SURF2/csi(8000)
   COMMON/SURFEKIN/uk(8000),sigekin(8000),Sk_Sk0(8000)
   DIMENSION yy(8000)
   external fcn
   u1=0.0d0
   DO l=1, nfn-1
      wwww=w(l)
      estr2=w(l)
!     IF(w(l).le.0.0d0)THEN
!     wwww=w(nfn-1)/10.d0
!     estr2=wwww
!     ENDIF
      call intgau(fcn,u1)
      uk(l)=dexp(wwww)*u1
   ENDDO
   sum=0.0d0
   b=0.0d0
   h=0.0d0
   xiR=0.0d0
   DO i=1,nfn
      xiR=csi(i)
      DO k=i,nfn
         yy(k)=dsqrt((csi(k)**2.d0)-(xiR**2.d0))
         IF(k==i)THEN
            sum=0.0d0
         ELSE
            b=uk(k)+uk(k-1)
            h=yy(k)-yy(k-1)
            sum=sum+b*h*0.5d0
!               PRINT*,sum,b,h
         ENDIF
      ENDDO
      IF(i==1)THEN
         sigekin0=sum
         IF(sum==0)THEN
            PRINT*,'zero central Skin'
         ENDIF

      ENDIF
      sigekin(i)=sum!!*(1.0d0/(1.0d0+R*R))

      Sk_Sk0(i)=sigekin(i)/sigekin0
   END DO
   return
   end

!     subroutine per il calcolo della curva calorica
   SUBROUTINE caloric_curve()
   IMPLICIT REAL*8(A-H,O-Z)
   COMMON/RESULTS/X(8000),W(8000),DW(8000),RAPP(8000),NFN
   COMMON/PARFCN/WWWW
   COMMON/ESTREMI/ESTR1,ESTR2
   COMMON/VALINI/W0,RHO0
   COMMON/CV2/xMhat,xnhat(8000),x0Cv(10),xi0Cv(10),potch(8000)
   COMMON/ENERGY/xk(8000),eg(8000),ephi(8000),xkt,egt,ephit,Etot,Vir2&
   ,et_r(8000),Etot1
   COMMON/CC/yEt_Mv02(10000),xs2_v02(10000),fw0,xsave,ysave,w0save
   DIMENSION xs(10000),ys(10000),xre(10000)
   real, DIMENSION(:), allocatable:: xre2
   real, DIMENSION(:), allocatable:: xsv
   real, DIMENSION(:), allocatable:: ysv
!      PRINT*,'calcolo curva calorica'
!     y=x*f(w0)
   fw0=Etot/xMhat

!      PRINT*,fw0,w0
   xincr=7.0d0/1.d+4
   yEt_Mv02(1)=-2.0d0
   xincr2= 1.5d0/1.0d+4
   xs2_v02(1)=0.5d0
   xs(1)=0.0d0
   ys(1)=0.0d0
   DO i=2,10000
      xs2_v02(i)=xs2_v02(i-1)+xincr2        !\in[0.5,2]
      yEt_Mv02(i)=yEt_Mv02(i-1)+xincr !\in[-2,5]
   ENDDO
!     Per un dato w0, c'� un solo valore di y e di x tc y=xf(w0).
!     E' dopo che nel grafico (x,y) trovo punti con diverso w0 ma stella y o x
   xsave=0.0d0
   ysave=0.d0
   w0save=0.d0
   k=0
   DO i=1,10000
      xxx=0.0d0
      xxx=xs2_v02(i)
      ex2=0.0d0
      yyy=0.0d0
      xxxx=0.0d0

      DO j=1,10000
         yyy=yEt_Mv02(j)
         fw0=Etot/xMhat
         xxxx=yyy/fw0
         IF((abs((xxxx-xxx)/xxx)).lt.(abs((xs(i)-xxx)/xxx)).AND.&
         (abs((xxxx-xxx)/xxx).le.1.0d-7))THEN
            k=k+1
            xs(k)=xxxx
            ys(k)=yyy
            XRE(k)=abs((xs(k)-xxx)/xxx)
            ex2=1.0d0
         ENDIF
      ENDDO
   ENDDO

   allocate (xre2(1:k))
   allocate (xsv(1:k))
   allocate (ysv(1:k))
   DO i=1,k
      xre2(i)=xre(i)
      xsv(i)=xs(i)
      ysv(i)=ys(i)
   ENDDO

   xresave=MINVAL(xre2)          !xs(k)
!      PRINT*,'xsave=',xsave
   DO i=1,k
      IF(xresave-xre2(i).eq.0.0d0)THEN
         xsave=xsv(i)
         ysave=ysv(i)
         w0save=w0
         PRINT*,'x=',xsave,'y=',ysave
      ENDIF

   ENDDO

!!      DO i=1,10000
!         xxx=xs2_v02(i)



   RETURN
   END

!     Funzione per il calcolo della velocit� quadratica media
   function fcnqd2(t)
   IMPLICIT REAL*8 (A-H,O-Z)
   COMMON/PARFCN/WWWW
   fcnqd2 = (dexp(wwww-t))*(t**2.5d0)
   return
   end

!     subroutine per il calcolo della velocit� quadratica media
   SUBROUTINE v2mean(fcn1,fcn2)
   IMPLICIT REAL*8(A-H,O-Z)
   COMMON/RESULTS/X(8000),W(8000),DW(8000),RAPP(8000),NFN
   COMMON/VEL/v2(8000)
   COMMON/PARFCN/WWWW
   COMMON/ESTREMI/ESTR1,ESTR2
   EXTERNAL fcn1,fcn2
         DO l=1, nfn-1
         wwww=w(l)
         estr2=w(l)
!           IF(w(l).le.0.0d0)THEN
!              wwww=w(nfn-1)/10.d0
!              estr2=wwww
!           ENDIF
         call intgau(fcn1,qd)
         v2_winf=qd
         call intgau(fcn2,qd2)
         v2(l)=(0.4d0)*(qd2/v2_winf)
!           PRINT*,qd, qd2
      ENDDO
      v2(nfn)=v2(nfn-1)
      return
      end


!     subroutine per il calcolo della densit� superficiale
   SUBROUTINE surfdens(fcn)
   IMPLICIT REAL*8(A-H,O-Z)
   COMMON/RESULTS/X(8000),W(8000),DW(8000),RAPP(8000),NFN
   COMMON/SURF/t(8000),phi(8000),sigma(8000),psi(8000),S_S0(8000)
   COMMON/SURF2/csi(8000)
   COMMON/ESTREMI/estr1,estr2
   COMMON/PARFCN/WWWW
   DIMENSION yy(8000)
   EXTERNAL fcn
   qd=0.0d+0
   DO l=1,nfn-1
      wwww=w(l)
      estr2=wwww
      call intgau(fcn,qd)
      psi(l)=qd              !anche psi(l)=rpp*sigma0
   END DO


   sum=0.0d0
   b=0.0d0
   h=0.0d0
   xiR=0.0d0
   DO i=1,nfn
      xiR=csi(i)
      DO k=i,nfn
         yy(k)=dsqrt((csi(k)**2.d0)-(xiR**2.d0))
         IF(k==i)THEN
            sum=0.0d0
         ELSE
            b=psi(k)+psi(k-1)
            h=yy(k)-yy(k-1)
            sum=sum+b*h*0.5d0
!               PRINT*,sum,b,h
         ENDIF
      ENDDO
      IF(i==1)THEN
         sigma0=sum
      ENDIF
      sigma(i)=sum!*(1.0d0/(1.0d0+R*R))

      s_s0(i)=sigma(i)/sigma0
   END DO



!      sum=0.0d+0
!      DO k=1, nfn-1           !nfn-1
!         R=csi(k)
!         DO l=k,nfn-1
!            t(l)=(csi(l)*csi(l)-R*R)/(1+csi(l)*csi(l))
!            phi(l)=((1+csi(l)*csi(l))**(1.5d0))*psi(l)
!            IF(l==k)THEN
!               sum=0.0d0
!               phi0=phi(l)
!               t0=t(l)          !che � zero
!            ELSEIF(l==k+1)THEN
!               t1=t(l)
!               t2=(t0+t1)/2.0d0 ! t1/2
!               phi1=phi(l)
!!               phi2=(phi1+phi0)/2.0d0
!               phi2=phi0/((1.0d0+t1/2.0d0)**(1.5d0))
!               sum=sum+(4.d+0/15.d+0)*dsqrt(t1)*((5.d+0-(t1/t2))*phic0
!     1             + phi1*(3.0d0*t1-5.0d0*t2)/(2.0d0*(t1-t2))
!     1              + phi2*(t1*t1)/(t2*(t1-t2)))
!            ELSE
!            b=phi(l)*(1.0d0/dsqrt(t(l)))+phi(l-1)*(1.0d0/dsqrt(t(l-1)))
!               h=t(l)-t(l-1)
!               sum=sum+b*h*0.5d+0
!            ENDIF
!         END DO
!         IF(k==1)sigma0=sum
!         sigma(k)=sum*(1.0d0/(1.0d0+R*R))
!
!         s_s0(k)=sigma(k)/sigma0
!      END DO

   return
   end

!     Subroutine per il calcolo delle energie totali adimensionali
   SUBROUTINE energie(f1,f3,fcn1)
   IMPLICIT REAL*8(A-H,O-Z)
   COMMON/ESTREMI/ESTR1,ESTR2
   COMMON/VALINI/W0,RHO0
   COMMON/GAUSS/INDGAU
   COMMON/COSTANTI/PAI
   COMMON/PARFCN/WWWW
   COMMON/RESULTS/X(8000),W(8000),DW(8000),RAPP(8000),NFN
   COMMON/SURF2/csi(8000)
   COMMON/PARFU2/qq
   COMMON/ENERGY/xk(8000),eg(8000),ephi(8000),xkt,egt,ephit,Etot,Vir2,&
   et_r(8000),Etot1
   COMMON/Mrho0hat/rho0hat,xMcap,xmu,xnhat0,conc
   DIMENSION rhat(8000)
   EXTERNAL f1,f3, fcn1
   z=0.0d0
   u1=0.0d0
   u3=0.0d0
   DO m=1,nfn-1
      z=w(m)
      wwww=w(m)
      estr2=wwww
      call intgau(f1,u1)
      wwww=w(m)
      estr2=wwww
      call intgau(f3,u3)
      xk(m)=dexp(z)*u1*4.d0*pai*dsqrt(2.d0)
!         PRINT*,u1,m,w(m)
      eg(m)=(2.d0/3.d0)*(dexp(z)*u1+0.4d0*(z**(2.5d0)))*dw(m)*x(m)*&
      4.d0*pai*dsqrt(2.d0)
      IF(abs(u3).gt.1.d-30)THEN
         ephi(m)=-xk(m)-dexp(z)*u3*4.d0*pai*dsqrt(2.d0) !-dexp(z)*(u1+u3)
      ELSE
         ephi(m)=0.0d0
         ENDIF
      et_r(m)=xk(m)+eg(m)+ephi(m)
!         PRINT*,ephi(m),u3,w(m),m!,eg(m)
   ENDDO
   xk(nfn)=0.0d0
   eg(nfn)=0.0d0
   ephi(nfn)=0.0d0
   et_r(nfn)=0.0d0

   wwww=w0
   estr2=w0
   qd=0.0d0
   call intgau(fcn1,qd)
   xnhat0=qd
   rho0hat=(4.d0*dsqrt(2.d0)*PAI)*qd!xnhat(1) !xnhat(1)=xnhat(w0)
   xMcap=((9.d0/(4.d0*PAI))**(1.5d0))*(1.d0/dsqrt(rho0hat))*xmu
   DO i=1,nfn
      rhat(i)=dsqrt(9.0d0/(4.d0*PAI))*(1.d0/dsqrt(rho0hat))*x(i)
   ENDDO


   sum1=0.0d0
   sum2=0.0d0
   sum3=0.0d0
   b1=0.0d0
   b2=0.0d0
   b3=0.0d0
   h=0.0d0
   DO k=1,nfn-1
!         b1=xk(k+1)*csi(k+1)*csi(k+1)+xk(k)*csi(k)*csi(k)
!         b2=eg(k+1)*csi(k+1)*csi(k+1)+eg(k)*csi(k)*csi(k)
!         b3=ephi(k+1)*csi(k+1)*csi(k+1)+ephi(k)*csi(k)*csi(k)
!     h=csi(k+1)-csi(k)
      b1=xk(k+1)*rhat(k+1)*rhat(k+1)+xk(k)*rhat(k)*rhat(k)
      b2=eg(k+1)*rhat(k+1)*rhat(k+1)+eg(k)*rhat(k)*rhat(k)
      b3=ephi(k+1)*rhat(k+1)*rhat(k+1)+ephi(k)*rhat(k)*rhat(k)
      h=rhat(k+1)-rhat(k)
      sum1=sum1+b1*h*0.5d0
      sum2=sum2+b2*h*0.5d0
      sum3=sum3+b3*h*0.5d0
!         PRINT*,sum2
   ENDDO
   xkt=0.0d0
   egt=0.0d0
   ephit=0.0d0
   xkt=sum1*4.d0*pai!!((9.d0/(4.d0*pai*rho0hat))**(1.5d0))*
!     1     (conc**3.d0)
   egt=sum2*4.d0*pai!*((9.d0/(4.d0*pai*rho0hat))**(1.5d0))*
!     1     (conc**3.d0)
   ephit=sum3*4.d0*pai!*((9.d0/(4.d0*pai*rho0hat))**(1.5d0))*
!     1     (conc**3.d0)
   Etot=0.0d0
   Etot=(sum1+sum2+sum3)
   Etot1=xkt+egt+ephit
   Vir=2.d0*xkt+egt              !dovrebbe essere nullo
!      PRINT*,'2Ktot+Egr=',Vir,2.d0*xkt,egt
   Vir2=2.d0*xkt/(-egt)          !dovrebbe essere 1
!      PRINT*,'2Ktot/-Egr=',Vir2
!      PRINT*,Etot,sum1,sum2,sum3



   return
   end

!     Subroutine per il calcolo del calore specifico adimensionale

   SUBROUTINE calorespecifico(fcn1,f1,f2,f3,f4)!,funzerr)
   IMPLICIT REAL*8(A-H,O-Z)
   COMMON/ESTREMI/ESTR1,ESTR2
   COMMON/VALINI/W0,RHO0
   COMMON/GAUSS/INDGAU
   COMMON/COSTANTI/PAI
   COMMON/PARFCN/WWWW
   COMMON/RESULTS/X(8000),W(8000),DW(8000),RAPP(8000),NFN
   COMMON/SURF2/csi(8000)
   COMMON/CV/Cv0(8000),Cv(8000),Cvqt(8000),Ctot0,Ctot1,Ctot2
   COMMON/CV2/xMhat,xnhat(8000),x0Cv(10),xi0Cv(10),potch(8000)
   COMMON/PARFU2/qq
   COMMON/TEST/C,D
   COMMON/Mrho0hat/rho0hat,xMcap,xmu,xnhat0,conc
   EXTERNAL fcn1,f1,f2,f3,f4,funzerr
   DO k=1,nfn
      Cv0(k)=0.0d0
      Cv(k)=0.0d0
      Cvqt(k)=0.0d0
      xnhat(k)=0.0d0
   ENDDO


   DO m=1,nfn-1
      z=0.0d0
      zz=0.0d0
      wwww=w(m)
      estr2=wwww
      z=w(m)

      qd=0.0d0
      u1=0.0d0
      u21=0.0d0
      u23=0.0d0
      u3=0.0d0
      u4=0.0d0
      zz=dsqrt(z)
!     call intgau(funzerr,zzz)
!         zz=derf(dsqrt(z))
!         qd=(dsqrt(PAI)/2.0d0)*dexp(z)*(derf(zz))-(z**(0.5d0))*(1.0d0
!     1        +(2.0d0*z/3.0d0))
      call intgau(fcn1,qd)
!         PRINT*,qd1-qd,derf(zz)
!         u1=dexp(-z)*(1.5d0*qd-(2.0d0/5.0d0)*(z**(2.5d0)))
!         u21=dexp(-z)*(1.5d0*qd-(2.0d0/3.0d0)*(z**(2.5d0)))
!         u23=dexp(-z)*(3.75d0*qd-(z**(2.5d0))*(1.0d0+0.4d0*z))

!         u21=u1-(4.d0/15.d0)*dexp(-z)*(z**(2.5d0))
!         u23=2.5d0*u1-0.4d0*dexp(-z)*(z**(3.5d0))
!         call intgau(fcnqd1,qd)
!     PRINT*,qd,qd1,qd2,qd-qd1
      xnhat(m)=qd!*2.0d0*pai
      wwww=w(m)
      estr2=wwww
      call intgau(f1,u1)
!
!     qq=1/2 oppure 3/2 a seconda di cosa compare in fu2
      wwww=w(m)
      estr2=wwww
      qq=0.5d0
      call intgau(f2,u21)

      qq=1.5d0
      wwww=w(m)
      estr2=wwww
      call intgau(f2,u23)
!     PRINT*,u2_3,u2_33,u2_3-u2_33

      potch(m)=u21*dexp(z)/qd

      wwww=w(m)
      estr2=wwww
      call intgau(f3,u3)

!         test=wwww*(0.9d0)
!         estr1=0.0d0
!         estr2=test
!         call intgau(fu3,u3_)
!         PRINT*,u3,u3_,u3-u3_
      wwww=w(m)
      estr1=0.0d0
      estr2=wwww
      call intgau(f4,u4)


!      Cv0(m)=dexp(wwww)*(u3*(1.5d0-((2.0d0/3.0d0)*(wwww**(2.5d0))/qd))
!     1        -u4)
      Cv0(m)=(((u3*u21*dexp(2.0d0*z))/qd)-u4*dexp(z))
!         PRINT*,u3


      Cv(m)=(Cv0(m)+x(m)*dw(m)*(qd+(2.0d0/3.0d0)*&
      (z**(1.5d0))*(2.5d0-1.4d0*z)+((4.0d0/9.0d0)*(z**(4.0d0))*&
      (0.4d0*z-2.0d0)/qd)+ ((16.d0/135.d0)*(z**(6.5d0))/(qd*qd))))

!      Cvqt(m)=2*PAI*(Cv0(m)-3.0d0*qd-2.0d0*(wwww**(1.5d0))+
!     1     +4.0d0*(wwww**(3.5d0))/5.0d0
!     1     +8.0d0*(wwww**(4.0d0))/(9.0d0*qd))

!     Cvqt(m)=(Cv0(m)-2.d0*dexp(z)*u23+(2.d0*dexp(2.d0*z)*u1*u21/qd))

      gw=dexp(z)*erf(zz)*(-z+(dsqrt(2.d0)-1.d0)/2.d0)-&
      (z+dsqrt(2.d0*z))/dsqrt(pai)
      hw=(zz/4.d0+dsqrt(pai)*gw/4.d0+(z**(1.5d0))/2.d0)/&
      ( -zz/2.d0+dexp(z)*dsqrt(pai)*erf(zz)/4.d0-(z**(1.5d0))/3.d0)

!         Cvqt(m)=(Cv0(m)+0.5d0*(1.5d0-C+(dsqrt(z)/4.d0
!     1        -dsqrt(pai)*dexp(z)*z*erf(dsqrt(z))/4.d0-z/2.d0+
!     1        (z**(1.5d0))/2.d0)/(-dsqrt(z)/4.d0+dsqrt(pai)*dexp(z)*
!     1 derf(dsqrt(z))/4.d0-(z**(1.5d0))/3.d0 ))*(-C-z)*xnhat(m))
      Cvqt(m)=(Cv0(m)+0.5d0*(1.5d0-C+hw)*(-C-z)*xnhat(m))

!      PRINT*,Cv0(m),Cv(m),Cvqt(m),xnhat(m),wwww

!      IF(w(1).eq.0.7d0)THEN
!         PRINT*,'problema strano'
!         PRINT*,w(m),qd,u1,u2_1,u2_3,u3,u4
!      ENDIF

   ENDDO
   wwww=w0
   estr2=w0
   qd=0.0d0
   call intgau(fcn1,qd)
   xnhat0=qd
   rho0hat=(4.d0*dsqrt(2.d0)*PAI)*qd!xnhat(1) !xnhat(1)=xnhat(w0)
   xMcap=((9.d0/(4.d0*PAI))**(1.5d0))*(1.d0/dsqrt(rho0hat))*xmu

!      PRINT*,xMcap
   qd=0.0d0
   xnhat(nfn)=0.0d0
   u1=0.0d0
   u21=0.0d0
   u23=0.0d0
   u3=0.0d0
   u4=0.0d0
   Cv0(nfn)=0.0d0
   Cv(nfn)=0.0d0
   Cvqt(nfn)=0.0d0

   sum0=0.0d0
   sum1=0.0d0
   sum2=0.0d0
   sum3=0.0d0
   b0=0.0d0
   b1=0.0d0
   b2=0.0d0
   b3=0.0d0
   h=0.0d0
   DO k=1,nfn-1
      b0=(Cv0(k+1)*csi(k+1)*csi(k+1)+Cv0(k)*csi(k)*csi(k))
      b1=(Cv(k+1)*csi(k+1)*csi(k+1)+Cv(k)*csi(k)*csi(k))
      b2=(Cvqt(k+1)*csi(k+1)*csi(k+1)+Cvqt(k)*csi(k)*csi(k))
      b3=(xnhat(k+1)*csi(k+1)*csi(k+1)+xnhat(k)*csi(k)*csi(k))
      h=csi(k+1)-csi(k)
      sum0=sum0+b0*h*0.5d0
      sum1=sum1+b1*h*0.5d0
      sum2=sum2+b2*h*0.5d0
      sum3=sum3+b3*h*0.5d0
   ENDDO
   Ctot0=0.0d0
   Ctot1=0.0d0
   Ctot2=0.0d0
   Ctot0=sum0/sum3
   Ctot1=sum1/sum3
   Ctot2=sum2/sum3

   xMhat=sum3
   DO i=1,10
      x0Cv(i)=0.0d0
      xi0Cv(i)=0.0d0
   ENDDO

   ex=0.0d0
   ex2=0.0d0
   ex3=0.0d0
   DO m=1,nfn-1
      IF((Cv(m).ge.0.0d0).AND.(Cv(m+1).le.0.0d0)&
         .AND.(ex.eq.0.0d0))THEN
         x0Cv(1)=(x(m)+x(m+1))*0.5d0
         xi0Cv(1)=(csi(m)+csi(m+1))*0.5d0
         ex=1.0d0
!            PRINT*,'primo zero Cv: r/rk=',x0Cv(1),'r/rt=',xi0Cv(1),
!     1           'w0=', w0,m
      ELSEIF((Cv(m).le.0.0d0).AND.(Cv(m+1).ge.0.0d0)&
         .AND.(ex2.eq.0.0d0))THEN
         x0Cv(2)=(x(m)+x(m+1))*0.5d0
         xi0Cv(2)=(csi(m)+csi(m+1))*0.5d0
         ex2=1.0d0
!            PRINT*,'secondo zero Cv: r/rk=',x0Cv(2),'r/rt=',xi0Cv(2),
!     1           'w0=', w0,m
      ELSEIF((Cv(m).ge.0.0d0).AND.(Cv(m+1).le.0.0d0)&
         .AND.(ex.eq.1.0d0).AND.(ex3.eq.0.0d0))THEN
            x0Cv(3)=(x(m)+x(m+1))*0.5d0
            xi0Cv(3)=(csi(m)+csi(m+1))*0.5d0
            ex3=1.0d0
!               PRINT*,'terzo zero Cv: r/rk=',x0Cv(3),'r/rt=',xi0Cv(3),
!     1              'w0=', w0,m
!            GOTO 9999
      ENDIF
   ENDDO
 9999 CONTINUE

   RETURN
   END

   function funzerr(x)
   IMPLICIT REAL*8 (A-H,O-Z)
!      COMMON/PARFCN/WWWW
   funzerr= dexp(-(x**2.0d0))
   return
   end
!    la funzione da integrare (ce ne possono essere pi� di una)
!
!
   function fcnqd(x)! result(f)
   IMPLICIT REAL*8 (A-H,O-Z)
!   real*8 :: f,x
   COMMON/PARFCN/WWWW
   fcnqd = (dexp(wwww-x))*(x**1.5d0)
   return
   end

   function fcnqd1(x)
   IMPLICIT REAL*8 (A-H,O-Z)
   COMMON/PARFCN/WWWW
   fcnqd1=(dexp(wwww-x)-1.0d0)*(x**(0.5d0))
   return
   end

   function fu1(x)
   IMPLICIT REAL*8 (A-H,O-Z)
   COMMON/PARFCN/WWWW
   fu1 = (dexp(-x)-dexp(-wwww))*(x**1.5d0)
   return
   end

   function fu2(x)
   IMPLICIT REAL*8 (A-H,O-Z)
   COMMON/PARFCN/WWWW
   COMMON/PARFU2/qq
   fu2 = (x*dexp(-x)-wwww*dexp(-wwww))*(x**(qq))
   return
   end

   function fu3(x)
   IMPLICIT REAL*8(A-H,O-Z)
   COMMON/PARFCN/WWWW
   fu3=(dexp(-x)-dexp(-wwww))*dlog(dexp(-x)-dexp(-wwww))*(x**(0.5d0))
   return
   end

   function fu4(x)
   IMPLICIT REAL*8(A-H,O-Z)
   COMMON/PARFCN/WWWW
   fu4=(x*dexp(-x)-wwww*dexp(-wwww))*dlog(dexp(-x)-dexp(-wwww))*&
   (x**(0.5d0))
   return
   end
!
!    la subroutine con la equazione differenziale
!
    SUBROUTINE FCT (X,Y,DERY, FCN)
    IMPLICIT REAL*8 (A-H,O-Z)
   COMMON/PARFCN/WWWW
   COMMON/ESTREMI/ESTR1,ESTR2
   COMMON/VALINI/W0,RHO0
   COMMON/COSTANTI/PAI
    COMMON/FCTOUTP/RPP
    DIMENSION Y(2),DERY(2)
   EXTERNAL :: fcn
   wwww = y(1)
   estr2= y(1)

   if(wwww.gt.0.d0) go to 1
   rpp = 0.d0
   go to 2
1      continue
   call intgau(fcn,qd)
   rpp = qd/rho0
2      continue

    dery(1) = y(2)
    dery(2) = -(2.d0/x)*y(2)-9.d0*rpp

    return
    end


!    la subroutine dei valori di output per l'integrazione
!
    SUBROUTINE OUTP (XX,Y,DERY,IHLF,NDIM,PRMT)
    IMPLICIT REAL*8(A-H,O-Z)
    DIMENSION Y(2),DERY(2),PRMT(5)
   COMMON/RESULTS/X(8000),W(8000),DW(8000),RAPP(8000),NFN
    COMMON/FCTOUTP/RPP

    if (y(1).gt.0.d0) goto 2

 1      PRMT(5)=1.d0
        goto 3

 2      xrif = x(nfn) + prmt(3)
        if(xx.lt.xrif) return

 3      nfn=nfn+1
    if (nfn.eq.8000) PRINT*,'TROPPI PASSI'
    if (nfn.eq.8000) PRMT(5)=1.00

    x(nfn)  = xx
        w(nfn)  = y(1)
      dw(nfn)  = y(2)
     rapp(nfn)  = rpp

    return
    end

!    la subroutine per il calcolo degli integrali
!
   subroutine INTGAU(fcn,area)
!
!       INTEGRA CON IL METODO DI GAUSS-LEGENDRE LA FUNZIONE FCN TRA A E B
!       I PARAMETRI DELLA FCN SONO PASSATI CON UN COMMON TRA PROGRAMMA
!       CHIAMANTE LA SUBROUTINE INTGAU E LA FUNCTION FCN(X).
!       I LIMITI DI INTEGRAZIONE CON UN COMMON TRA PROGRAMMA CHIAMANTE E
!       LE SUBROUTINES GAUS10.FOR, GAUS20.FOR, GAUS40.FOR, GAUS80.FOR.
!
!
    IMPLICIT REAL*8 (A-H,O-Z)
   real*8, external :: fcn
   common/gauss/indgau
   goto (1,2,3,4) ,indgau
 1    call GAUS20 (fcn,area)
   return
 2    call GAUS40 (fcn,area)
   return
 3    call GAUS80 (fcn,area)
   return
4     call GAUS96 (fcn,area)
   return

    end

!
!    la subroutine dhpcg che � il cuore del programma


      SUBROUTINE DHPCG(PRMT,Y,DERY,NDIM,IHLF,FCT,OUTP,AUX,FCN)
      IMPLICIT REAL*8 (A-H,O-Z)
      DIMENSION PRMT(5),Y(2),DERY(2),AUX(16,2)
      EXTERNAL fcn
      N=1
      IHLF=0
      X=PRMT(1)
      H=PRMT(3)
      PRMT(5)=0.D0
      DO 1 I=1,NDIM
         AUX(16,I)=0.D0
         AUX(15,I)=DERY(I)
         AUX(1,I)=Y(I)
 1    CONTINUE
      IF(H*(PRMT(2)-X).lt.0.0d0)THEN
         GOTO 3
      ELSE IF(H*(PRMT(2)-X).eq.0.0d0)THEN
         GOTO 2
      ELSE
         GOTO 4
      ENDIF
 2    IHLF=12
      GOTO 4
 3    IHLF=13
 4    CONTINUE
      CALL FCT(X,Y,DERY,fcn)
      CALL OUTP (X,Y,DERY,IHLF,NDIM,PRMT)
      IF((PRMT(5).lt.0.0d0).OR.(PRMT(5).gt.0.0d0))THEN
        GOTO 6
      ELSE
         GOTO 5
      ENDIF

 5    IF(IHLF.le.0.0d0)THEN
        GOTO 7
      ELSE
         GOTO 6
      ENDIF
 6    RETURN
 7    DO 8 I=1,NDIM
         AUX(8,I)=DERY(I)
 8    CONTINUE
      ISW=1
      GOTO 100
 9    X=X+H
      DO 10 I=1,NDIM
         AUX(2,I)=Y(I)
 10   CONTINUE
 11   IHLF=IHLF+1
      X=X-H
      DO 12 I=1,NDIM
         AUX(4,I)=AUX(2,I)
 12   CONTINUE
      H=.5D0*H
      N=1
      ISW=2
      GOTO 100
 13   X=X+H
      CALL FCT(X,Y,DERY,fcn)
      N=2
      DO 14 I=1,NDIM
         AUX(2,I)=Y(I)
         AUX(9,I)=DERY(I)
 14   CONTINUE
      ISW=3
      GOTO 100
 15   DELT=0.D0
      DO 16 I=1,NDIM
         DELT=DELT+AUX(15,I)*DABS(Y(I)-AUX(4,I))
 16   CONTINUE
      DELT=.066666666666666667D0*DELT
      IF((DELT-PRMT(4)).le.0.0d0)THEN
         GOTO 19
      ELSE
         GOTO 17
      ENDIF

 17   IF((IHLF-10).lt.0.0d0)THEN
         GOTO 11
      ELSE
         GOTO 18
      ENDIF

 18   IHLF=11
      X=X+H
      GOTO 4
 19   X=X+H
      CALL FCT(X,Y,DERY,fcn)
      DO 20 I=1,NDIM
         AUX(3,I)=Y(I)
         AUX(10,I)=DERY(I)
 20   CONTINUE
      N=3
      ISW=4
      GOTO 100
 21   N=1
      X=X+H
      CALL FCT(X,Y,DERY,fcn)
      X=PRMT(1)
      DO 22 I=1,NDIM
    AUX(11,I)=DERY(I)
    Y(I)=AUX(1,I)+H*(.375D0*AUX(8,I)+.7916666666666667D0*AUX(9,I)&
      -.20833333333333333D0*AUX(10,I)+.041666666666666667D0*DERY(I))
 22   CONTINUE
 23   X=X+H
      N=N+1
      CALL FCT(X,Y,DERY,fcn)
      CALL OUTP (X,Y,DERY,IHLF,NDIM,PRMT)
      IF((PRMT(5).lt.0.0d0).OR.(PRMT(5).gt.0.0d0))THEN
         GOTO 6
      ELSE
         GOTO 24
      ENDIF

 24   IF((N-4).lt.0.0d0)THEN
         GOTO 25
      ELSE
         GOTO 200
      ENDIF

 25   DO 26 I=1,NDIM
         AUX(N,I)=Y(I)
         AUX(N+7,I)=DERY(I)
 26   CONTINUE
      IF((N-3).lt.0.0d0)THEN
         GOTO 27
      ELSEIF( (N-3).eq.0.0d0)THEN
         GOTO 29
      ELSE
         GOTO 200
      ENDIF
 27   DO 28 I=1,NDIM
    DELT=AUX(9,I)+AUX(9,I)
    DELT=DELT+DELT
        Y(I)=AUX(1,I)+.33333333333333333D0*H*(AUX(8,I)+DELT+AUX(10,I))
 28   CONTINUE
      GOTO 23
 29   DO 30 I=1,NDIM
         DELT=AUX(9,I)+AUX(10,I)
         DELT=DELT+DELT+DELT
         Y(I)=AUX(1,I)+.375D0*H*(AUX(8,I)+DELT+AUX(11,I))
 30   CONTINUE
      GOTO 23
 100  DO 101 I=1,NDIM
         Z=H*AUX(N+7,I)
         AUX(5,I)=Z
         Y(I)=AUX(N,I)+.4D0*Z
 101  CONTINUE
      Z=X+.4D0*H
      CALL FCT(Z,Y,DERY,fcn)
      DO 102 I=1,NDIM
         Z=H*DERY(I)
         AUX(6,I)=Z
      Y(I)=AUX(N,I)+.29697760924775360D0*AUX(5,I)+.15875964497103583D0*Z
 102  CONTINUE
    Z=X+.45573725421878943D0*H
    CALL FCT(Z,Y,DERY,fcn)
    DO 103 I=1,NDIM
    Z=H*DERY(I)
    AUX(7,I)=Z
    Y(I)=AUX(N,I)+.21810038822592047D0*AUX(5,I)-3.0509651486929308D0*&
            AUX(6,I)+3.8328647604670103D0*Z
 103  CONTINUE
      Z=X+H
      CALL FCT(Z,Y,DERY,fcn)
      DO 104 I=1,NDIM
       Y(I)=AUX(N,I)+.17476028226269037D0*AUX(5,I)-.55148066287873294D0*&
       AUX(6,I)+1.2055355993965235D0*AUX(7,I)+.17118478121951903D0*&
       H*DERY(I)
 104  CONTINUE
      GOTO(9,13,15,21),ISW
 200  ISTEP=3
 201  IF(((N-8).lt.0.0d0).OR.((N-8).gt.0.0d0))THEN
         GOTO 204
      ELSE
         GOTO 202
      ENDIF

 202  DO 203 N=2,7
         DO  I=1,NDIM
            AUX(N-1,I)=AUX(N,I)
            AUX(N+6,I)=AUX(N+7,I)
         ENDDO
 203  CONTINUE
         N=7
 204     N=N+1
         DO 205 I=1,NDIM
            AUX(N-1,I)=Y(I)
            AUX(N+6,I)=DERY(I)
 205     CONTINUE
         X=X+H
 206     ISTEP=ISTEP+1
         DO 207 I=1,NDIM
    DELT=AUX(N-4,I)+1.3333333333333333D0*H*(AUX(N+6,I)+AUX(N+6,I)-&
    AUX(N+5,I)+AUX(N+4,I)+AUX(N+4,I))
    Y(I)=DELT-.9256198347107438D0*AUX(16,I)
        AUX(16,I)=DELT
 207    CONTINUE
    CALL FCT(X,Y,DERY,fcn)
    DO 208 I=1,NDIM
    DELT=.125D0*(9.D0*AUX(N-1,I)-AUX(N-3,I)+3.D0*H*(DERY(I)+AUX(N+6,I)&
    +AUX(N+6,I)-AUX(N+5,I)))
    AUX(16,I)=AUX(16,I)-DELT
        Y(I)=DELT+.07438016528925620D0*AUX(16,I)
 208    CONTINUE
    DELT=0.D0
    DO 209 I=1,NDIM
           DELT=DELT+AUX(15,I)*DABS(AUX(16,I))
 209    CONTINUE
        IF((DELT-PRMT(4)).lt.0.0d0)THEN
           GOTO 210
        ELSE
           GOTO 222
        ENDIF

210    CONTINUE
        CALL FCT(X,Y,DERY,fcn)
        CALL OUTP (X,Y,DERY,IHLF,NDIM,PRMT)
    IF((PRMT(5).lt.0.0d0).OR.(PRMT(5).gt.0.0d0))THEN
           GOTO 212
        ELSE
           GOTO 211
        ENDIF
 211    IF((IHLF-11).lt.0.0d0)THEN
           GOTO 213
        ELSE
           GOTO 212
        ENDIF
 212    RETURN
 213    IF(H*(X-PRMT(2)).lt.0.0d0)THEN
           GOTO 214
        ELSE
           GOTO 212
        ENDIF
 214    IF(DABS(X-PRMT(2))-.1D0*DABS(H).lt.0.0d0)THEN
           GOTO 212
        ELSE
           GOTO 215
        ENDIF
 215    IF(DELT-.02D0*PRMT(4).le.0.0d0)THEN
           GOTO 216
        ELSE
           GOTO 201
        ENDIF
 216    IF(IHLF.le.0.0d0)THEN
           GOTO 201
        ELSE
           GOTO 217
        ENDIF
 217    IF(N-7.lt.0.0d0)THEN
           GOTO 201
        ELSE
           GOTO 218
        ENDIF
 218    IF(ISTEP-4.lt.0.0d0)THEN
           GOTO 201
        ELSE
           GOTO 219
        ENDIF
 219    IMOD=ISTEP/2
    IF(ISTEP-IMOD-IMOD.ne.0.0d0)THEN
           GOTO 201
        ELSE
           GOTO 220
        ENDIF
220    H=H+H
    IHLF=IHLF-1
    ISTEP=0
    DO 221 I=1,NDIM
    AUX(N-1,I)=AUX(N-2,I)
    AUX(N-2,I)=AUX(N-4,I)
    AUX(N-3,I)=AUX(N-6,I)
    AUX(N+6,I)=AUX(N+5,I)
    AUX(N+5,I)=AUX(N+3,I)
    AUX(N+4,I)=AUX(N+1,I)
    DELT=AUX(N+6,I)+AUX(N+5,I)
    DELT=DELT+DELT+DELT
    AUX(16,I)=8.962962962962963D0*(Y(I)-AUX(N-3,I))&
    -3.3611111111111111D0*H*(DERY(I)+DELT+AUX(N+4,I))
 221    CONTINUE
        GOTO 201
222    IHLF=IHLF+1
    IF(IHLF-10.le.0.0d0)THEN
           GOTO 223
        ELSE
           GOTO 210
        ENDIF
 223    H=.5D0*H
    ISTEP=0
    DO 224 I=1,NDIM
    Y(I)=.390625D-2*(8.D1*AUX(N-1,I)+135.D0*AUX(N-2,I)+4.D1*AUX(N-3,I)&
    +AUX(N-4,I))-.1171875D0*(AUX(N+6,I)-6.D0*AUX(N+5,I)-AUX(N+4,I))*H
    AUX(N-4,I)=.390625D-2*(12.D0*AUX(N-1,I)+135.D0*AUX(N-2,I)+&
    108.D0*AUX(N-3,I)+AUX(N-4,I))-.0234375D0*(AUX(N+6,I)+&
    18.D0*AUX(N+5,I)-9.D0*AUX(N+4,I))*H
    AUX(N-3,I)=AUX(N-2,I)
        AUX(N+4,I)=AUX(N+5,I)
 224    CONTINUE
    X=X-H
    DELT=X-(H+H)
    CALL FCT(DELT,Y,DERY,fcn)
    DO 225 I=1,NDIM
    AUX(N-2,I)=Y(I)
    AUX(N+5,I)=DERY(I)
        Y(I)=AUX(N-4,I)
 225    CONTINUE
    DELT=DELT-(H+H)
    CALL FCT(DELT,Y,DERY,fcn)
    DO 226 I=1,NDIM
    DELT=AUX(N+5,I)+AUX(N+4,I)
    DELT=DELT+DELT+DELT
    AUX(16,I)=8.962962962962963D0*(AUX(N-1,I)-Y(I))&
    -3.3611111111111111D0*H*(AUX(N+6,I)+DELT+DERY(I))
        AUX(N+3,I)=DERY(I)
 226    CONTINUE
    GOTO 206
    END
