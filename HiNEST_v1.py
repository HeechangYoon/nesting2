import csv
import os
import numpy as np
import time
import copy
from datetime import datetime, timedelta
import matplotlib.pyplot as plt


class Schedule():
    ID = int(0)

    def __init__(ss, Dock, ShipNo, LOT, ProcessingBay, ProcessingStage, ProcessingStartDate, ProcessingEndDate,
                 ShipType):
        ss.ID = Schedule.ID
        ss.Dock = Dock
        ss.ShipNo = ShipNo
        ss.LOT = LOT
        ss.ProcessingBay = ProcessingBay
        ss.ProcessingStage = ProcessingStage
        ss.ProcessingStartDate = datetime.strptime(ProcessingStartDate, '%Y%m%d').date()
        ss.ProcessingEndDate = datetime.strptime(ProcessingEndDate, '%Y%m%d').date()
        ss.ShipType = ShipType
        Schedule.ID = Schedule.ID + 1


class Part():
    ID = int(0)

    def __init__(qq, ShipNo, Block, TribonName, Type, Symmetry, P, S, Grade, Area, Thick, LOT, WorkingStage, FAB):

        qq.ID = Part.ID

        # from part list
        qq.ShipNo = ShipNo
        qq.Block = Block
        qq.TribonName = TribonName
        qq.Type = Type
        qq.Symmetry = Symmetry
        qq.P = P
        qq.S = S
        qq.Grade = Grade
        qq.AreaFromPartList = float(Area) * 1.0e+6  # m2 --> mm2
        qq.Thick = float(Thick)
        qq.LOT = LOT
        qq.WorkingStage = WorkingStage
        qq.FAB = FAB

        # from genfile list
        qq.GenFilePath = []
        qq.GenFileName = []

        # from compiling
        qq.WebHeight = []
        qq.InitialNodeID = []
        qq.ContoursMarking, qq.ContoursBurning = [], []
        qq.ContoursMargin = []
        qq.Area, qq.UCOR, qq.VCOR, qq.UCOG, qq.VCOG, qq.IuuCOG, qq.IuvCOG, qq.IvvCOG, qq.IwwCOG = [], [], [], [], [], [], [], [], []
        qq.DotUMarking, qq.DotVMarking, qq.GDotUMarking, qq.GDotVMarking = [], [], [], []
        qq.DotUBurning, qq.DotVBurning, qq.GDotUBurning, qq.GDotVBurning = [], [], [], []
        qq.DotUMargin, qq.DotVMargin, qq.GDotUMargin, qq.GDotVMargin = [], [], [], []
        qq.TB, qq.SB = [], []
        qq.DotUSurfPoint, qq.DotVSurfPoint = [], []
        qq.CDotUMarking, qq.CDotVMarking = [], []
        qq.CDotUBurning, qq.CDotVBurning = [], []
        qq.CDotUMargin, qq.CDotVMargin = [], []
        qq.CDotUSurfPoint, qq.CDotVSurfPoint = [], []
        qq.AngleDegList, qq.PixelPart, qq.PixelMargin = [], [], []

        # from allocation
        qq.FinalNodeID = []
        qq.PlateID = []
        qq.AllocationX, qq.AllocationY, qq.AllocationA = [], [], []
        qq.AllocationIdxW, qq.AllocationIdxH, qq.AllocationIdxA = [], [], []

        Part.ID = Part.ID + 1

    def Update_Schedule(qq, sList):
        # qq.Update_Schedule(ss): 생산 일정을 부재 리스트에 업데이트하는 메서드
        for ss in sList:
            if qq.ShipNo == ss.ShipNo and qq.LOT + 'L' == ss.LOT:
                qq.Dock = ss.Dock
                qq.ProcessingBay = ss.ProcessingBay
                qq.ProcessingStage = ss.ProcessingStage
                qq.ProcessingStartDate = ss.ProcessingStartDate
                qq.ProcessingEndDate = ss.ProcessingEndDate
                qq.ShipType = ss.ShipType
                qq.ScheduleID = ss.ID

    def Update_GenFileName(qq, GenFilePath, GenFileNameList, GenFileExtn):
        # qq.Update_GenFileName(GenFilePath,GenFileNameList): gen 파일명을 부재리스트에 업데이트하는 메서드
        TribonName0 = qq.TribonName + GenFileExtn
        TribonName1 = TribonName0.replace('/', '_')
        TribonName2 = TribonName0.replace('/', '-')
        GenFileName = []
        if len([i for i in range(len(GenFileNameList)) if GenFileNameList[i] == TribonName0]) > 0:
            GenFileName = TribonName0
        elif len([i for i in range(len(GenFileNameList)) if GenFileNameList[i] == TribonName1]) > 0:
            GenFileName = TribonName1
        elif len([i for i in range(len(GenFileNameList)) if GenFileNameList[i] == TribonName2]) > 0:
            GenFileName = TribonName2
        else:
            GenFileName = 'NA'
        qq.GenFileName = GenFileName
        qq.GenFilePath = GenFilePath

    def IsIncludedToNode(qq, nn):
        # qq.isIncludedToNode(nn): 부재 qq가 노드 nn에 해당되는지 True/False를 리턴하는 메서드
        Condition = True
        if qq.ShipNo != nn.ShipNo:              Condition = Condition * False  # 호선번호
        if qq.Block != nn.Block:               Condition = Condition * False  # 블록
        if qq.ProcessingStage != nn.ProcessingStage:     Condition = Condition * False  # 가공작업장
        if qq.ProcessingStartDate != nn.ProcessingStartDate: Condition = Condition * False  # 가공착수일
        if qq.Grade != nn.Grade:               Condition = Condition * False  # 재질
        if qq.Thick != nn.Thick:               Condition = Condition * False  # 두께
        if qq.FAB != nn.FAB:                 Condition = Condition * False  # 계열
        if qq.LOT != nn.LOT:                 Condition = Condition * False  # 로트
        return Condition

    def Update_InitialNode(qq, nList):
        # nList = qq.Update_InitialNode(nList): 부재 qq의 원소속 노드를 업데이트하고 노드 리스트를 리턴하는 메서드
        if qq.GenFileName == 'NA':
            return nList
        else:
            isThereCompatibleNode = 0  # 0:No, 1:Yes
            for nn in nList:
                if qq.IsIncludedToNode(nn):
                    isThereCompatibleNode = 1
                    nn.Qn = nn.Qn + [qq.ID]
                    qq.InitialNodeID = nn.ID
            if isThereCompatibleNode == 0:
                nn = Node(qq)
                nn.Qn = nn.Qn + [qq.ID]
                nList = nList + [nn]
                qq.InitialNodeID = nn.ID
            return nList

    def Gen2MarkingAndBurningContour(qq):
        # 2022.09.26: stiffener의 gen 파일 읽어올 수 있도록 기능 추가함.
        genFile = open(qq.GenFilePath + '\\' + qq.GenFileName, 'r')
        contoursMarking, contourMarking = [], []
        contoursBurning, contourBurning = [], []
        webHeight = []
        if qq.Type == 'stiffener':
            # stiffener인 경우
            flagCommonData, flagLeftEnd, flagRightEnd, flagGeneralMarkingData = False, False, False, False
            contourLE, contourRE, contourGM, contoursGM = [], [], [], []
            leftOriginAtLE, leftOriginAtRE, distOriginAtGM = [], [], []
            while True:
                genLine = genFile.readline().rstrip('\n')
                if not genLine: break
                # stiffener의 web height 읽어오기(나중에 stiffener 배치 시 필요한 정보임)
                if flagCommonData == True:
                    if "WEB_HEIGHT=" == genLine[0:11]: webHeight = float(genLine[11:])
                    if "END_OF_COMMON_DATA" == genLine[0:18]: flagCommonData = False
                if "COMMON_DATA" == genLine[0:11]: flagCommonData = True
                # stiffener의 left end 형상 정보 읽어오기
                if flagLeftEnd == True:
                    if "LEFT_ORIGIN=" == genLine[0:12]: leftOriginAtLE = float(genLine[12:])
                    if "START_U=" == genLine[0:8]:  contourLE.append(float(genLine[8:]) + leftOriginAtLE)
                    if "START_V=" == genLine[0:8]:  contourLE.append(float(genLine[8:]))
                    if "AMP_U=" == genLine[0:6]:  contourLE.append(float(genLine[6:]))
                    if "AMP_V=" == genLine[0:6]:  contourLE.append(float(genLine[6:]))
                    if "AMP=" == genLine[0:4]:  contourLE.append(float(genLine[4:]))
                    if "SWEEP=" == genLine[0:6]:  contourLE.append(float(genLine[6:]))
                    if "U=" == genLine[0:2]:  contourLE.append(float(genLine[2:]) + leftOriginAtLE)
                    if "V=" == genLine[0:2]:  contourLE.append(float(genLine[2:]))
                    if "END_OF_LEFT_END" == genLine[0:15]: flagLeftEnd = False
                if "LEFT_END" == genLine[0:8]: flagLeftEnd = True
                # stiffener의 right end 형상 정보 읽어오기
                if flagRightEnd == True:
                    if "LEFT_ORIGIN=" == genLine[0:12]: leftOriginAtRE = float(genLine[12:])
                    if "START_U=" == genLine[0:8]:  contourRE.append(-float(genLine[8:]) + leftOriginAtRE)
                    if "START_V=" == genLine[0:8]:  contourRE.append(float(genLine[8:]))
                    if "AMP_U=" == genLine[0:6]:  contourRE.append(-float(genLine[6:]))
                    if "AMP_V=" == genLine[0:6]:  contourRE.append(float(genLine[6:]))
                    if "AMP=" == genLine[0:4]:  contourRE.append(float(genLine[4:]))
                    if "SWEEP=" == genLine[0:6]:  contourRE.append(float(genLine[6:]))
                    if "U=" == genLine[0:2]:  contourRE.append(-float(genLine[2:]) + leftOriginAtRE)
                    if "V=" == genLine[0:2]:  contourRE.append(float(genLine[2:]))
                    if "END_OF_RIGHT_END" == genLine[0:16]: flagRightEnd = False
                if "RIGHT_END" == genLine[0:9]: flagRightEnd = True
                # stiffener의 marking data 읽어오기
                if flagGeneralMarkingData == True:
                    if "DIST_ORIGIN=" == genLine[0:12]: distOriginAtGM = float(genLine[12:])
                    if "TYPE=" == genLine[0:5]:  contourGM.append(genLine[5:])
                    if "START_U=" == genLine[0:8]:  contourGM.append(float(genLine[8:]) + distOriginAtGM)
                    if "START_V=" == genLine[0:8]:  contourGM.append(float(genLine[8:]))
                    if "AMP_U=" == genLine[0:6]:  contourGM.append(float(genLine[6:]))
                    if "AMP_V=" == genLine[0:6]:  contourGM.append(float(genLine[6:]))
                    if "AMP=" == genLine[0:4]:  contourGM.append(float(genLine[4:]))
                    if "SWEEP=" == genLine[0:6]:  contourGM.append(float(genLine[6:]))
                    if "U=" == genLine[0:2]:  contourGM.append(float(genLine[2:]) + distOriginAtGM)
                    if "V=" == genLine[0:2]:  contourGM.append(float(genLine[2:]))
                    if "END_OF_GENERAL_MARKING_DATA" == genLine[0:27]:
                        flagGeneralMarkingData = False
                        contoursGM = contoursGM + [contourGM]
                        contourGM = []
                if "GENERAL_MARKING_DATA" == genLine[0:20]: flagGeneralMarkingData = True

            def calculateArc(Ui, Vi, ampU, ampV, Uj, Vj):
                # Ui,Vi,ampU,ampV,Uj,Vj로부터 arc의 원점(Uo,Vo) 및 반경(R) 계산하여 리턴
                Uk, Vk = (Ui + Uj) / 2 + ampU, (Vi + Vj) / 2 + ampV
                K = np.array([[Ui, Vi, 1], [Uj, Vj, 1], [Uk, Vk, 1]], dtype=float)
                b = np.array([[-Ui ** 2 - Vi ** 2], [-Uj ** 2 - Vj ** 2], [-Uk ** 2 - Vk ** 2]])
                x = np.linalg.inv(K) @ b
                Uo, Vo, R = -x[0, 0] / 2, -x[1, 0] / 2, np.sqrt((x[0, 0] / 2) ** 2 + (x[1, 0] / 2) ** 2 - x[2, 0])
                return Uo, Vo, R

            # Left end 정보 포멧 맞춰주기
            noSegmentLE = int((len(contourLE) - 2) / 6)
            contourBurningLE = []
            for s in range(noSegmentLE):
                Ui, Vi = contourLE[6 * s + 0], contourLE[6 * s + 1]
                ampU, ampV, amp = contourLE[6 * s + 2], contourLE[6 * s + 3], contourLE[6 * s + 4]
                sweep = contourLE[6 * s + 5]
                Uj, Vj = contourLE[6 * s + 6], contourLE[6 * s + 7]
                Uo, Vo, R = 0., 0., 0.
                if sweep > 1.e-9: Uo, Vo, R = calculateArc(Ui, Vi, ampU, ampV, Uj, Vj)
                if (Ui - Uj) ** 2 + (Vi - Vj) ** 2 > 1.e-3: contourBurningLE = contourBurningLE + [Ui, Vi, ampU, ampV,
                                                                                                   amp, R, sweep, Uo,
                                                                                                   Vo]
            contourBurningLE = contourBurningLE + [contourLE[-2], contourLE[-1]]

            # Right end 정보 포멧 맞춰주기. 단, 현재는 시계방향임.
            noSegmentRE = int((len(contourRE) - 2) / 6)
            contourBurningREinCW = []
            for s in range(noSegmentRE):
                Ui, Vi = contourRE[6 * s + 0], contourRE[6 * s + 1]
                ampU, ampV, amp = contourRE[6 * s + 2], contourRE[6 * s + 3], contourRE[6 * s + 4]
                sweep = contourRE[6 * s + 5]
                Uj, Vj = contourRE[6 * s + 6], contourRE[6 * s + 7]
                Uo, Vo, R = 0., 0., 0.
                if sweep > 1.e-9: Uo, Vo, R = calculateArc(Ui, Vi, ampU, ampV, Uj, Vj)
                if (Ui - Uj) ** 2 + (Vi - Vj) ** 2 > 1.e-3: contourBurningREinCW = contourBurningREinCW + [Ui, Vi, ampU,
                                                                                                           ampV, amp, R,
                                                                                                           sweep, Uo,
                                                                                                           Vo]
            contourBurningREinCW = contourBurningREinCW + [contourRE[-2], contourRE[-1]]

            # Right end 정보를 반시계방향으로 정렬.
            noSegmentRE = int((len(contourBurningREinCW) - 2) / 8)
            contourBurningRE = []
            for s in range(noSegmentRE):
                ampU = contourBurningREinCW[(noSegmentRE - (s + 1)) * 9 + 2]
                ampV = contourBurningREinCW[(noSegmentRE - (s + 1)) * 9 + 3]
                amp = contourBurningREinCW[(noSegmentRE - (s + 1)) * 9 + 4]
                R = contourBurningREinCW[(noSegmentRE - (s + 1)) * 9 + 5]
                sweep = contourBurningREinCW[(noSegmentRE - (s + 1)) * 9 + 6]
                Uo = contourBurningREinCW[(noSegmentRE - (s + 1)) * 9 + 7]
                Vo = contourBurningREinCW[(noSegmentRE - (s + 1)) * 9 + 8]
                Uj = contourBurningREinCW[(noSegmentRE - (s + 1)) * 9 + 9]
                Vj = contourBurningREinCW[(noSegmentRE - (s + 1)) * 9 + 10]
                contourBurningRE = contourBurningRE + [Uj, Vj, ampU, ampV, amp, R, sweep, Uo, Vo]
            contourBurningRE = contourBurningRE + [contourBurningREinCW[0], contourBurningREinCW[1]]

            # Left end 및 Right end 정보 엮어서 Closed loop 생성하기
            contoursBurning = ['OUTER_CONTOUR']
            contoursBurning = contoursBurning + contourBurningLE + [0., 0., 0., 0., 0., 0., 0.]
            contoursBurning = contoursBurning + contourBurningRE + [0., 0., 0., 0., 0., 0., 0.]
            contoursBurning = contoursBurning + [contourBurningLE[0], contourBurningLE[1]]
            contoursBurning = [contoursBurning]

            # General marking 정보 포멧 맞춰주기
            noContourGM = int(len(contoursGM))
            for g in range(noContourGM):
                contourGM = contoursGM[g]
                noSegmentGM = int((len(contourGM) - 3) / 6)
                contourMarking = []
                for s in range(noSegmentGM):
                    if s == 0:
                        markingType = contourGM[6 * s + 0]
                        contourMarking = contourMarking + [markingType]
                    Ui, Vi = contourGM[6 * s + 1], contourGM[6 * s + 2]
                    ampU, ampV, amp = contourGM[6 * s + 3], contourGM[6 * s + 4], contourGM[6 * s + 5]
                    sweep = contourGM[6 * s + 6]
                    Uj, Vj = contourGM[6 * s + 7], contourGM[6 * s + 8]
                    Uo, Vo, R = 0., 0., 0.
                    if sweep > 1.e-9: Uo, Vo, R = calculateArc(Ui, Vi, ampU, ampV, Uj, Vj)
                    contourMarking = contourMarking + [Ui, Vi, ampU, ampV, amp, R, sweep, Uo, Vo]
                contourMarking = contourMarking + [contourGM[-2], contourGM[-1]]
                contoursMarking = contoursMarking + [contourMarking]

        else:
            # stiffener가 아닌 경우
            flagMarking, flagBurning, flagContour, flagGeometry = False, False, False, False
            while True:
                genLine = genFile.readline().rstrip('\n')
                if not genLine: break
                # marking 데이터 읽어오기
                if flagMarking == True:
                    if "MARKING_TYPE=" == genLine[0:13]: contourMarking.append(genLine[13:])
                    if flagGeometry == False:
                        if flagContour == True:
                            if "START_U=" == genLine[0:8]: contourMarking.append(float(genLine[8:]))
                            if "START_V=" == genLine[0:8]: contourMarking.append(float(genLine[8:]))
                            if "AMP_U=" == genLine[0:6]: contourMarking.append(float(genLine[6:]))
                            if "AMP_V=" == genLine[0:6]: contourMarking.append(float(genLine[6:]))
                            if "AMP=" == genLine[0:4]: contourMarking.append(float(genLine[4:]))
                            if "RADIUS=" == genLine[0:7]: contourMarking.append(float(genLine[7:]))
                            if "SWEEP=" == genLine[0:6]: contourMarking.append(float(genLine[6:]))
                            if "ORIGIN_U=" == genLine[0:9]: contourMarking.append(float(genLine[9:]))
                            if "ORIGIN_V=" == genLine[0:9]: contourMarking.append(float(genLine[9:]))
                            if "U=" == genLine[0:2]: contourMarking.append(float(genLine[2:]))
                            if "V=" == genLine[0:2]: contourMarking.append(float(genLine[2:]))
                            if "END_OF_CONTOUR" == genLine[0:14]:
                                flagContour = False
                                contoursMarking = contoursMarking + [contourMarking]
                                contourMarking = []
                        if "START_OF_CONTOUR" == genLine[0:16]: flagContour = True
                    if "GEOMETRY_DATA" == genLine[0:13]: flagGeometry = True
                    if "END_OF_GEOMETRY_DATA" == genLine[0:20]: flagGeometry = False
                    if "END_OF_MARKING_DATA" == genLine[0:19]: flagMarking = False
                if "MARKING_DATA" == genLine[0:12]: flagMarking = True
                # burning 데이터 읽어오기
                if flagBurning == True:
                    if "SHAPE=" == genLine[0:6]: contourBurning.append(genLine[6:])
                    if flagGeometry == False:
                        if flagContour == True:
                            if "START_U=" == genLine[0:8]: contourBurning.append(float(genLine[8:]))
                            if "START_V=" == genLine[0:8]: contourBurning.append(float(genLine[8:]))
                            if "AMP_U=" == genLine[0:6]: contourBurning.append(float(genLine[6:]))
                            if "AMP_V=" == genLine[0:6]: contourBurning.append(float(genLine[6:]))
                            if "AMP=" == genLine[0:4]: contourBurning.append(float(genLine[4:]))
                            if "RADIUS=" == genLine[0:7]: contourBurning.append(float(genLine[7:]))
                            if "SWEEP=" == genLine[0:6]: contourBurning.append(float(genLine[6:]))
                            if "ORIGIN_U=" == genLine[0:9]: contourBurning.append(float(genLine[9:]))
                            if "ORIGIN_V=" == genLine[0:9]: contourBurning.append(float(genLine[9:]))
                            if "U=" == genLine[0:2]: contourBurning.append(float(genLine[2:]))
                            if "V=" == genLine[0:2]: contourBurning.append(float(genLine[2:]))
                            if "END_OF_CONTOUR" == genLine[0:14]:
                                flagContour = False
                                contoursBurning = contoursBurning + [contourBurning]
                                contourBurning = []
                        if "START_OF_CONTOUR" == genLine[0:16]: flagContour = True
                    if "GEOMETRY_DATA" == genLine[0:13]: flagGeometry = True
                    if "END_OF_GEOMETRY_DATA" == genLine[0:20]: flagGeometry = False
                    if "END_OF_BURNING_DATA" == genLine[0:19]: flagBurning = False
                if "BURNING_DATA" == genLine[0:12]: flagBurning = True

        genFile.close()
        qq.ContoursMarking = contoursMarking
        qq.ContoursBurning = contoursBurning
        qq.WebHeight = webHeight

    def ContourBurning2ContourMargin(qq):
        # burning contour로부터 margin contour를 생성하기
        contours = qq.ContoursBurning
        noContours = len(contours)
        idxContourType = 0
        idxStartU, idxStartV, idxAmpU, idxAmpV, idxAmp, idxRadius, idxSweep, idxOriginU, idxOriginV, idxEndU, idxEndV = np.arange(
            1, 11 + 1, 1)
        contoursMargin = []
        marginValue = 9. / 2
        for c in range(noContours):
            contour = contours[c]
            contourType = contour[idxContourType]
            contourMargin = []
            noSegment = int((len(contour) - 3) / 9)
            for s in range(noSegment):
                startU = contour[idxStartU + s * 9];
                startV = contour[idxStartV + s * 9]
                ampU = contour[idxAmpU + s * 9];
                ampV = contour[idxAmpV + s * 9];
                amp = contour[idxAmp + s * 9]
                radius = contour[idxRadius + s * 9];
                sweep = contour[idxSweep + s * 9]
                originU = contour[idxOriginU + s * 9];
                originV = contour[idxOriginV + s * 9]
                endU = contour[idxEndU + s * 9];
                endV = contour[idxEndV + s * 9]
                if ampU == 0. and ampV == 0. and amp == 0. and radius == 0. and sweep == 0. and originU == 0. and originV == 0.:
                    # (Ui,Vi)와 (Uj,Vj) 사이가 직선인 경우
                    startP, endP = np.array([startU, startV, 0]), np.array([endU, endV, 0])
                    P = endP - startP
                    p = P / np.linalg.norm(P)
                    k = np.array([0., 0., 1.])
                    q = np.cross(p, k)
                    startUMargin, startVMargin = startU + marginValue * q[0], startV + marginValue * q[1]
                    endUMargin, endVMargin = endU + marginValue * q[0], endV + marginValue * q[1]
                    segmentMargin = [contourType, startUMargin, startVMargin, 0., 0., 0., 0., 0., 0., 0., endUMargin,
                                     endVMargin]
                else:
                    # (Ui,Vi)와 (Uj,Vj) 사이가 Arc인 경우
                    O = np.array([originU, originV])
                    S = np.array([startU, startV])
                    E = np.array([endU, endV])
                    A = np.array([ampU, ampV])
                    O2S = S - O
                    O2E = E - O
                    O2A = (S + E) / 2 + A - O
                    O2newS = (radius + np.sign(amp) * marginValue) * O2S / radius
                    O2newE = (radius + np.sign(amp) * marginValue) * O2E / radius
                    O2newA = (radius + np.sign(amp) * marginValue) * O2A / radius
                    newS = O2newS + O;
                    newE = O2newE + O;
                    newA = O + O2newA - (newS + newE) / 2
                    newAmp = np.sign(amp) * np.sqrt(newA[0] ** 2 + newA[0] ** 2)
                    newRadius = radius + np.sign(amp) * marginValue
                    segmentMargin = [contourType, newS[0], newS[1], newA[0], newA[1], newAmp, newRadius, sweep, O[0],
                                     O[1], newE[0], newE[1]]
                contourMargin = contourMargin + [segmentMargin]
            contoursMargin = contoursMargin + contourMargin
        qq.ContoursMargin = contoursMargin

    def Contour2Dot(qq, contours, pixelSize):

        pixelSize = min(10., pixelSize)

        # contour 정보를 dot 정보로 변환하기
        noContours = len(contours)
        idxContourType = 0
        idxStartU, idxStartV, idxAmpU, idxAmpV, idxAmp, idxRadius, idxSweep, idxOriginU, idxOriginV, idxEndU, idxEndV = np.arange(
            1, 11 + 1, 1)
        groupedDotU, groupedDotV = [], []
        for c in range(noContours):
            contour = contours[c]
            contourType = contour[idxContourType]
            contourDotU, contourDotV = [], []
            noSegment = int((len(contour) - 3) / 9)
            for s in range(noSegment):
                startU = contour[idxStartU + s * 9];
                startV = contour[idxStartV + s * 9]
                ampU = contour[idxAmpU + s * 9];
                ampV = contour[idxAmpV + s * 9];
                amp = contour[idxAmp + s * 9]
                radius = contour[idxRadius + s * 9];
                sweep = contour[idxSweep + s * 9]
                originU = contour[idxOriginU + s * 9];
                originV = contour[idxOriginV + s * 9]
                endU = contour[idxEndU + s * 9];
                endV = contour[idxEndV + s * 9]
                segmentDotU, segmentDotV = [], []
                if sweep < 1.e-9:  #
                    L = np.sqrt((endU - startU) ** 2 + (endV - startV) ** 2)
                    noDot = int(np.floor(L / pixelSize)) + 5
                    segmentDotU = list(np.linspace(startU, endU, noDot))
                    segmentDotV = list(np.linspace(startV, endV, noDot))
                else:
                    L = radius * sweep
                    noDot = int(np.floor(L / pixelSize)) + 5
                    startW = np.arctan2(startV - originV, startU - originU);
                    endW = np.arctan2(endV - originV, endU - originU);
                    if startW < endW:
                        dotW1 = np.linspace(startW, endW, noDot)
                        dotW2 = np.linspace(endW, startW + 2 * np.pi, noDot)
                        dotW2 = np.flip(dotW2)  # start->end 순서를 유지하기 위해 어레이를 뒤집어야 함.
                    else:
                        dotW1 = np.linspace(endW, startW, noDot)
                        dotW1 = np.flip(dotW1)  # start->end 순서를 유지하기 위해 어레이를 뒤집어야 함.
                        dotW2 = np.linspace(startW, endW + 2 * np.pi, noDot)

                    ampUglobal, ampVglobal = ((startU + endU) / 2 + ampU), ((startV + endV) / 2 + ampV)
                    dotU1, dotV1 = (originU + radius * np.cos(dotW1)), (originV + radius * np.sin(dotW1))
                    dotU2, dotV2 = (originU + radius * np.cos(dotW2)), (originV + radius * np.sin(dotW2))
                    dotW = []
                    if min((dotU1 - ampUglobal) ** 2 + (dotV1 - ampVglobal) ** 2) < min(
                            (dotU2 - ampUglobal) ** 2 + (dotV2 - ampVglobal) ** 2):
                        dotW = dotW1
                    else:
                        dotW = dotW2
                    segmentDotU = list(originU + radius * np.cos(dotW))
                    segmentDotV = list(originV + radius * np.sin(dotW))
                contourDotU = contourDotU + segmentDotU
                contourDotV = contourDotV + segmentDotV
            groupedDotU = groupedDotU + [contourDotU]
            groupedDotV = groupedDotV + [contourDotV]
        allDotU = sum(groupedDotU, [])
        allDotV = sum(groupedDotV, [])
        return allDotU, allDotV, groupedDotU, groupedDotV

    def Dot2BoundingBox(qq, pixelSize):
        # 부재 dot 정보로부터 외접사각형(TB:Tight box) 및 정사각형(SB:Square box) 결정
        # 2022.09.26: Square box의 사이즈가 픽셀 사이즈의 홀수배가 되도록 수정함
        Umin, Umax = min(qq.DotUMargin), max(qq.DotUMargin)
        Vmin, Vmax = min(qq.DotVMargin), max(qq.DotVMargin)

        Ucenter, Vcenter = (Umin + Umax) / 2, (Vmin + Vmax) / 2

        dU, dV = Umax - Umin, Vmax - Vmin
        D = np.sqrt(dU ** 2 + dV ** 2)
        dimD = []
        if D % pixelSize == 0.:
            dimD = int(np.floor(D / pixelSize))
        else:
            dimD = int(np.floor(D / pixelSize)) + 1
        if dimD % 2 == 0:       dimD = dimD + 1  # 회전 중심이 존재하도록 홀수개로 바꿔줌
        dimD = dimD + 2  # dot를 회전할 때 범위를 초과하는 경우가 있어서 2만큼 증가
        D = dimD * pixelSize

        cUmin, cUmax = -D / 2, +D / 2
        cVmin, cVmax = -D / 2, +D / 2
        cUcenter, cVcenter = 0., 0.
        TB = {'Umin': Umin, 'Umax': Umax, 'Vmin': Vmin, 'Vmax': Vmax, 'Ucenter': Ucenter, 'Vcenter': Vcenter}
        SB = {'Umin': cUmin, 'Umax': cUmax, 'Vmin': cVmin, 'Vmax': cVmax, 'Ucenter': cUcenter, 'Vcenter': cVcenter}
        qq.TB = TB  # TB: tight(retangular) box
        qq.SB = SB  # SB: square box at zero-center

    def Dot2SurfPoint(qq, pixelSize):
        groupedDotUBurning = qq.GDotUBurning
        groupedDotVBurning = qq.GDotVBurning

        dotUBurning = qq.DotUBurning
        dotVBurning = qq.DotVBurning

        Umin, Umax = min(dotUBurning), max(dotUBurning)
        Vmin, Vmax = min(dotVBurning), max(dotVBurning)

        deltaU, deltaV = (Umax - Umin) / 5, (Vmax - Vmin) / 5
        deltaD = np.sqrt(deltaU ** 2 + deltaV ** 2)
        deltaU, deltaV = deltaD, deltaD

        noPointsOnSurf = 0
        dotUPointOnSurf, dotVPointOnSurf = [], []
        while noPointsOnSurf < 100:  # 100:
            noU = max(int(np.floor((Umax - Umin) / deltaU)) + 1, 4)
            noV = max(int(np.floor((Vmax - Vmin) / deltaV)) + 1, 4)
            noPoints = noU * noV
            tempU, tempV = np.linspace(Umin, Umax, noU), np.linspace(Vmin, Vmax, noV)
            gridU, gridV = np.meshgrid(tempU, tempV)
            gridU, gridV = gridU.reshape(-1, 1), gridV.reshape(-1, 1)

            # contour 별 integral dtheta 계산
            noContours = len(groupedDotUBurning)
            tableSumDtheta = np.empty([noPoints, noContours])
            for c in range(noContours):
                U1, V1 = np.array(groupedDotUBurning[c][:-1]), np.array(groupedDotVBurning[c][:-1])
                U2, V2 = np.array(groupedDotUBurning[c][1:]), np.array(groupedDotVBurning[c][1:])
                theta1 = np.arctan2(V1 - gridV, U1 - gridU) / np.pi * 180.
                theta2 = np.arctan2(V2 - gridV, U2 - gridU) / np.pi * 180.
                dtheta = theta2 - theta1
                dtheta = np.where(dtheta < -180., dtheta + 360., dtheta)
                dtheta = np.where(dtheta > +180., dtheta - 360., dtheta)
                sumDtheta = np.sum(dtheta, axis=1)
                tableSumDtheta[:, c] = sumDtheta

            # integral dtheta  값으로 포인트 위치가 표면인지 아닌지 판단
            threshold = 1.e-1
            isSurf, isSurfTemp = [1] * noPoints, [1] * noPoints
            for c in range(noContours):
                if c == noContours - 1:
                    isSurfTemp = np.where(np.abs(tableSumDtheta[:, c] - 360.) <= threshold, 1, 0)
                else:
                    isSurfTemp = np.where(np.abs(tableSumDtheta[:, c]) <= threshold, 1, 0)
                isSurf = isSurf * isSurfTemp

                # 표면 상의 포인트들만 취합
            idxPointOnSurf = np.where(isSurf == 1)[0].tolist()
            noPointsOnSurf = len(idxPointOnSurf)
            dotUPointOnSurf = gridU.reshape(-1)[idxPointOnSurf]
            dotVPointOnSurf = gridV.reshape(-1)[idxPointOnSurf]

            # 표면 상의 포인트들을 원하는 갯수 미만으로 찾으면 해상도 높여서 재시도
            deltaU, deltaV = deltaU / 2, deltaV / 2

        qq.DotUSurfPoint = dotUPointOnSurf
        qq.DotVSurfPoint = dotVPointOnSurf

    def Dot2Center(qq, dotU, dotV):
        Ucenter = qq.TB['Ucenter']
        Vcenter = qq.TB['Vcenter']
        cDotU, cDotV = dotU - Ucenter, dotV - Vcenter
        return cDotU, cDotV

    def DotRotating(qq, cDotU, cDotV, angleDeg):
        cDotUV = np.vstack((cDotU, cDotV))
        angleRad = angleDeg / 180 * np.pi
        rMat = np.array([[np.cos(angleRad), -np.sin(angleRad)], [np.sin(angleRad), np.cos(angleRad)]])
        rcDotUV = np.matmul(rMat, cDotUV)
        rcDotU, rcDotV = rcDotUV[0, :], rcDotUV[1, :]
        return rcDotU, rcDotV

    def BB2Dim(qq, pixelSize):
        Umin, Umax = qq.SB['Umin'], qq.SB['Umax']
        Vmin, Vmax = qq.SB['Vmin'], qq.SB['Vmax']

        dU, dV = Umax - Umin, Vmax - Vmin
        if dV % pixelSize == 0.:
            dimI = int(np.floor(dV / pixelSize))
        else:
            dimI = int(np.floor(dV / pixelSize)) + 1
        if dimI % 2 == 0: dimI = dimI + 1

        if dU % pixelSize == 0.:
            dimJ = int(np.floor(dV / pixelSize))
        else:
            dimJ = int(np.floor(dV / pixelSize)) + 1
        if dimJ % 2 == 0: dimJ = dimJ + 1

        return dimI, dimJ

    def Dot2Idx(qq, cDotU, cDotV, pixelSize):
        Umin, Umax = qq.SB['Umin'], qq.SB['Umax']
        Vmin, Vmax = qq.SB['Vmin'], qq.SB['Vmax']
        idxI = np.floor((Vmax - cDotV) / pixelSize)
        idxJ = np.floor((cDotU - Umin) / pixelSize)
        idxI = idxI.astype(int)
        idxJ = idxJ.astype(int)
        return idxI, idxJ

    def Dot2PixelContour(qq, cDotU, cDotV, pixelSize):
        dimI, dimJ = qq.BB2Dim(pixelSize)
        idxI, idxJ = qq.Dot2Idx(cDotU, cDotV, pixelSize)
        pixelMat = np.zeros((dimI, dimJ), dtype=int)
        pixelMat[idxI, idxJ] = 1
        return np.array(pixelMat, dtype=int)

    def PixelContour2PixelSurface(qq, pixelContour, pixelStartPoint):
        dimI, dimJ = pixelContour.shape
        pixelContour = np.array(pixelContour, dtype=int)
        pixelStartPoint = np.array(pixelStartPoint, dtype=int)
        pixelSurface = copy.deepcopy(pixelContour)

        startITempList, startJTempList = np.where(pixelStartPoint == 1)
        startIList, startJList = [], []
        for startIdxTemp in range(len(startITempList)):
            startITemp, startJTemp = startITempList[startIdxTemp], startJTempList[startIdxTemp]
            pixelState0 = (pixelSurface[startITemp, startJTemp] == 1)
            pixelState1 = (pixelSurface[startITemp - 1, startJTemp] == 1)
            pixelState2 = (pixelSurface[startITemp, startJTemp + 1] == 1)
            pixelState3 = (pixelSurface[startITemp + 1, startJTemp] == 1)
            pixelState4 = (pixelSurface[startITemp, startJTemp - 1] == 1)
            if not (pixelState0 or pixelState1 or pixelState2 or pixelState3 or pixelState4):
                startIList = startIList + [startITemp]
                startJList = startJList + [startJTemp]

        for startIdx in range(len(startIList)):
            startI, startJ = startIList[startIdx], startJList[startIdx]
            nowI, nowJ = [startI], [startJ]
            nextI, nextJ = [], []
            while True:
                if len(nowI) > 0:
                    for idx in range(len(nowI)):
                        if pixelSurface[nowI[idx], nowJ[idx]] == 0:
                            pixelSurface[nowI[idx], nowJ[idx]] = 1
                            if nowI[idx] - 1 >= 0 and pixelSurface[nowI[idx] - 1, nowJ[idx]] == 0:
                                nextI = nextI + [nowI[idx] - 1]
                                nextJ = nextJ + [nowJ[idx]]
                            if nowJ[idx] + 1 <= dimJ - 1 and pixelSurface[nowI[idx], nowJ[idx] + 1] == 0:
                                nextI = nextI + [nowI[idx]]
                                nextJ = nextJ + [nowJ[idx] + 1]
                            if nowI[idx] + 1 <= dimI - 1 and pixelSurface[nowI[idx] + 1, nowJ[idx]] == 0:
                                nextI = nextI + [nowI[idx] + 1]
                                nextJ = nextJ + [nowJ[idx]]
                            if nowJ[idx] - 1 >= 0 and pixelSurface[nowI[idx], nowJ[idx] - 1] == 0:
                                nextI = nextI + [nowI[idx]]
                                nextJ = nextJ + [nowJ[idx] - 1]
                    nowI, nowJ = nextI, nextJ
                    nextI, nextJ = [], []
                else:
                    break

        pixelSurface = np.where(pixelSurface + pixelStartPoint > 0, 1, 0) - pixelContour
        return np.array(pixelSurface, dtype=int)

    def Idx2CDot(qq, idxI, idxJ, pixelSize):
        Umin, Umax = qq.SB['Umin'], qq.SB['Umax']
        Vmin, Vmax = qq.SB['Vmin'], qq.SB['Vmax']
        Uhat = Umin + pixelSize * (idxJ + 0.5)
        Vhat = Vmax - pixelSize * (idxI + 0.5)
        return Uhat, Vhat

    def Dot2Pixel(qq, pixelSize, angleDegList, mainFcnName):
        dimI, dimJ = qq.BB2Dim(pixelSize)

        if mainFcnName == 'UpdatePixelStack':
            angleDegListTemp = []
            dUList = []
            dU = []
            for angleDeg in angleDegList:
                rcDotUBurning, rcDotVBurning = qq.DotRotating(qq.CDotUBurning, qq.CDotVBurning, angleDeg)
                dU = np.max(rcDotUBurning) - np.min(rcDotUBurning)
                if dU > 2 * 500.: angleDegListTemp = angleDegListTemp + [angleDeg]
                dUList = dUList + [dU]
            if len(angleDegListTemp) == 0: angleDegListTemp = angleDegList
            if qq.Type == 'stiffener': angleDegListTemp = [0, 180]
            qq.AngleDegList = angleDegListTemp
            qq.PixelPart = np.zeros((dimI, dimJ, len(qq.AngleDegList)), dtype=int)
            qq.PixelMargin = np.zeros((dimI, dimJ, len(qq.AngleDegList)), dtype=int)

        elif mainFcnName == 'UpdateGeometricProperty':
            angleDegListTemp = angleDegList

        for a, angleDeg in enumerate(angleDegListTemp):
            # cDot를 angleDeg로 회전하기
            rcDotUBurning, rcDotVBurning = qq.DotRotating(qq.CDotUBurning, qq.CDotVBurning, angleDeg)
            rcDotUMargin, rcDotVMargin = qq.DotRotating(qq.CDotUMargin, qq.CDotVMargin, angleDeg)
            rcDotUSurfPoint, rcDotVSurfPoint = qq.DotRotating(qq.CDotUSurfPoint, qq.CDotVSurfPoint, angleDeg)
            # 회전된 rcDot를 contour pixel 행렬로 변환하기
            rcPixelContourBurning = qq.Dot2PixelContour(rcDotUBurning, rcDotVBurning, pixelSize)
            rcPixelMargin = qq.Dot2PixelContour(rcDotUMargin, rcDotVMargin, pixelSize)
            rcPixelSurfPoint = qq.Dot2PixelContour(rcDotUSurfPoint, rcDotVSurfPoint, pixelSize)
            # cotour pixel 행렬을 채워서 surface pixel 행렬 만들기
            rcPixelSurfaceBurning = qq.PixelContour2PixelSurface(rcPixelContourBurning, rcPixelSurfPoint)
            # contour pixel과 surface pixel을 합쳐서 part pixel 만들기.
            # 부재의 테두리가 표시나도록 Burning contour의 픽셀 값은 크게 함.
            rcPixelPart = rcPixelSurfaceBurning + 2 * rcPixelContourBurning
            # 각도별 픽셀 이미지 쌓기
            if mainFcnName == 'UpdatePixelStack':
                qq.PixelPart[:, :, a] = np.array(rcPixelPart, dtype=int)
                qq.PixelMargin[:, :, a] = np.array(rcPixelMargin, dtype=int)
            elif mainFcnName == 'UpdateGeometricProperty':
                return rcPixelSurfaceBurning, rcPixelContourBurning

    def ReadGenFile(qq):
        qq.Gen2MarkingAndBurningContour()  # qq.ContoursMarking, qq.ContoursBurning
        qq.ContourBurning2ContourMargin()  # qq.ContoursMargin

    def UpdateGeometricProperty(qq):
        pixelSize = 10.  # 100.에서 20.으로 수정할 것
        angleDegList = [0.]
        qq.DotUBurning, qq.DotVBurning, qq.GDotUBurning, qq.GDotVBurning = qq.Contour2Dot(qq.ContoursBurning, pixelSize)
        qq.DotUMargin, qq.DotVMargin, qq.GDotUMargin, qq.GDotVMargin = qq.Contour2Dot(qq.ContoursMargin, pixelSize)
        qq.Dot2BoundingBox(pixelSize)  # qq.TB,qq.SB
        qq.Dot2SurfPoint(pixelSize)  # qq.DotUSurfPoint,qq.DotVSurfPoint
        qq.CDotUBurning, qq.CDotVBurning = qq.Dot2Center(qq.DotUBurning, qq.DotVBurning)
        qq.CDotUMargin, qq.CDotVMargin = qq.Dot2Center(qq.DotUMargin, qq.DotVMargin)
        qq.CDotUSurfPoint, qq.CDotVSurfPoint = qq.Dot2Center(qq.DotUSurfPoint, qq.DotVSurfPoint)
        pixelSurface, pixelContour = qq.Dot2Pixel(pixelSize, angleDegList, 'UpdateGeometricProperty')
        dA = pixelSize * pixelSize
        dASurface = 1.0 * dA
        dAContour = 0.5 * dA
        # Area
        Area = np.sum(pixelSurface) * dASurface + np.sum(pixelContour) * dAContour
        # Center of rotation = Origin
        UCOR, VCOR = 0., 0.
        # Center of gravity = UCOG,VCOG
        idxISurface, idxJSurface = np.where(pixelSurface)
        idxIContour, idxJContour = np.where(pixelContour)
        USurface, VSurface = qq.Idx2CDot(idxISurface, idxJSurface, pixelSize)
        UContour, VContour = qq.Idx2CDot(idxIContour, idxJContour, pixelSize)
        UCOG = 1 / Area * (np.sum(USurface) * dASurface + np.sum(UContour) * dAContour)
        VCOG = 1 / Area * (np.sum(VSurface) * dASurface + np.sum(VContour) * dAContour)
        # Ixx, Ixy, Iyy, Izz at COG UCOG,VCOG
        UCOG2Surface, VCOG2Surface = USurface - UCOG, VSurface - VCOG
        UCOG2Contour, VCOG2Contour = UContour - UCOG, VContour - VCOG
        IuuCOG = np.sum(VCOG2Surface ** 2) * dASurface + np.sum(VCOG2Contour ** 2) * dAContour
        IuvCOG = np.sum(UCOG2Surface * VCOG2Surface) * dASurface + np.sum(UCOG2Contour * VCOG2Contour) * dAContour
        IvvCOG = np.sum(UCOG2Surface ** 2) * dASurface + np.sum(UCOG2Contour ** 2) * dAContour
        IwwCOG = IuuCOG ** 2 + IvvCOG ** 2
        # Update attribute
        # qq.Area = Area
        qq.Area = qq.AreaFromPartList
        qq.UCOR, qq.VCOR = UCOR, VCOR
        qq.UCOG, qq.VCOG = UCOG, VCOG
        qq.IuuCOG, qq.IuvCOG, qq.IvvCOG, qq.IwwCOG = IuuCOG, IuvCOG, IvvCOG, IwwCOG

    def UpdatePixelStack(qq, pixelSize, angleDegList):
        qq.DotUMarking, qq.DotVMarking, qq.GDotUMarking, qq.GDotVMarking = qq.Contour2Dot(qq.ContoursMarking, pixelSize)
        qq.DotUBurning, qq.DotVBurning, qq.GDotUBurning, qq.GDotVBurning = qq.Contour2Dot(qq.ContoursBurning, pixelSize)
        qq.DotUMargin, qq.DotVMargin, qq.GDotUMargin, qq.GDotVMargin = qq.Contour2Dot(qq.ContoursMargin, pixelSize)
        qq.Dot2BoundingBox(pixelSize)  # qq.TB,qq.SB
        qq.Dot2SurfPoint(pixelSize)  # qq.DotUSurfPoint,qq.DotVSurfPoint
        qq.CDotUMarking, qq.CDotVMarking = qq.Dot2Center(qq.DotUMarking, qq.DotVMarking)
        qq.CDotUBurning, qq.CDotVBurning = qq.Dot2Center(qq.DotUBurning, qq.DotVBurning)
        qq.CDotUMargin, qq.CDotVMargin = qq.Dot2Center(qq.DotUMargin, qq.DotVMargin)
        qq.CDotUSurfPoint, qq.CDotVSurfPoint = qq.Dot2Center(qq.DotUSurfPoint, qq.DotVSurfPoint)
        qq.Dot2Pixel(pixelSize, angleDegList,
                     'UpdatePixelStack')  # qq.AngleDegList,qq.PixelPart,qq.PixelMargin

    def CompilePart(qq, pixelSize, angleDegList):
        if qq.GenFileName != 'NA':
            qq.ReadGenFile()  # qq.ContoursMarking,qq.ContoursBurning,qq.ContoursMargin 생성
            qq.UpdateGeometricProperty()  # qq.Area,qq.UCOR,qq.VCOR,qq.UCOG,qq.VCOG,qq.IuuCOG,qq.IuvCOG,qq.IvvCOG,qq.IwwCOG 계산
            qq.UpdatePixelStack(pixelSize, angleDegList)  # qq.AngleDegList, qq.PixelPart,qq.PixelMargin 생성

    def PlotDot(qq, figSizeValue=8):
        if qq.GenFileName != 'NA':
            # Dot 이미지 그리기
            dotUBB = [qq.SB['Umin'], qq.SB['Umax'], qq.SB['Umax'], qq.SB['Umin'], qq.SB['Umin']]
            dotVBB = [qq.SB['Vmin'], qq.SB['Vmin'], qq.SB['Vmax'], qq.SB['Vmax'], qq.SB['Vmin']]
            plt.figure(figsize=(figSizeValue, figSizeValue))
            plt.scatter(qq.CDotUMarking, qq.CDotVMarking)
            plt.scatter(qq.CDotUBurning, qq.CDotVBurning)
            plt.scatter(qq.CDotUMargin, qq.CDotVMargin)
            plt.scatter(qq.CDotUSurfPoint, qq.CDotVSurfPoint)
            plt.scatter(qq.UCOR, qq.VCOR, marker='o', s=figSizeValue * 20)
            plt.scatter(qq.UCOG, qq.VCOG, marker='+', s=figSizeValue * 20)
            plt.plot(dotUBB, dotVBB)
            plt.show()
        else:
            print('Gen File: NA')

    def PlotPixel(qq, figSizeValue=8):
        if qq.GenFileName != 'NA':
            # 부재 pixel 이미지 그리기
            pixelPart = np.array(qq.PixelPart[:, :, 0], dtype=int)
            plt.figure(figsize=(figSizeValue, figSizeValue))
            plt.imshow(pixelPart)
            # 마진 pixel 이미지 그리기
            # pixelMargin = np.array(qq.PixelMargin[:,:,0],dtype=int)
            # plt.figure(figsize=(figSizeValue,figSizeValue))
            # plt.imshow(pixelMargin)
        else:
            print('Gen File: NA')


class Node():
    ID = int(0)

    def __init__(nn, qq):
        nn.ID = Node.ID
        nn.ShipNo = qq.ShipNo
        nn.Block = qq.Block
        nn.Grade = qq.Grade
        nn.Thick = qq.Thick
        nn.LOT = qq.LOT
        nn.FAB = qq.FAB
        nn.ProcessingBay = qq.ProcessingBay
        nn.ProcessingStage = qq.ProcessingStage
        nn.ProcessingStartDate = qq.ProcessingStartDate
        nn.ProcessingEndDate = qq.ProcessingEndDate
        nn.ScheduleID = qq.ScheduleID
        nn.Qn = []
        nn.AllocatedQn = []
        Node.ID = Node.ID + 1

    def IsPossibleToGoToNode(nnFrom, nnTo, gDict, ShowProcess=False):
        ## nnFrom.IsPossibleToNode(nnTo,gDict,ShowFalseReason=False)
        # nnFrom 노드가 nnTo 노드로 합네스팅 가능한지 확인하는 함수
        # gDict: grade합 규칙을 딕셔너리 구조로 만든 것. key는 part 재질이고, value는 plate 재질.
        # fDict: FAB합 규칙을 딕셔너리 구조로 만들 것.
        Condition = True
        if not (nnFrom.ShipNo == nnTo.ShipNo):
            Condition = Condition * False
            if ShowProcess == True:
                print('. False : ShipNo')
        if not (nnFrom.Block == nnTo.Block):
            Condition = Condition * False
            if ShowProcess == True:
                print('. False : Block')
        if not (nnFrom.ProcessingStage == nnTo.ProcessingStage):
            Condition = Condition * False
            if ShowProcess == True:
                print('. False : ProcessingStage')
        Date1 = (nnTo.ProcessingStartDate - nnFrom.ProcessingStartDate).days
        Date2 = (nnFrom.ProcessingStartDate - nnTo.ProcessingStartDate).days
        if not (Date1 <= 0 and Date2 <= 8):
            Condition = Condition * False
            if ShowProcess == True:
                print('. False : ProcessingDate')
        if not (nnTo.Grade in gDict[nnFrom.Grade]):
            Condition = Condition * False
            if ShowProcess == True:
                print('. False : Grade')
        if not (nnFrom.Thick <= nnTo.Thick and nnTo.Thick <= nnFrom.Thick + 2.):
            Condition = Condition * False
            if ShowProcess == True:
                print('. False : Thick')
        if not (nnFrom.FAB == nnTo.FAB):  # TBU
            Condition = Condition * False
            if ShowProcess == True:
                print('. False : FAB')
        return Condition

    def IsPossibleToGoToPlate(nnFrom, ppTo, gDict, ShowProcess=False):
        ## nnFrom.IsPossibleToGoToPlate(ppTo,gDict,ShowFalseReason=False)
        # nnFrom 노드가 ppTo 자재로 합네스팅 가능한지 확인하는 함수
        # gDict: grade합 규칙을 딕셔너리 구조로 만든 것. key는 part 재질이고, value는 plate 재질.
        Condition = True
        if not (nnFrom.ProcessingStage == ppTo.ProcessingStage):
            Condition = Condition * False
            if ShowProcess == True:
                print('. False : ProcessingStage')
        Date1 = (ppTo.DateFrom - nnFrom.ProcessingStartDate).days  # <= 0 이면 합 가능
        Date2 = (nnFrom.ProcessingEndDate - ppTo.DateTo).days  # <= 0 이면 합 가능
        if not (Date1 <= 0 and Date2 <= 0):
            Condition = Condition * False
            if ShowProcess == True:
                print('. False : ProcessingDate')
        if not (ppTo.Grade in gDict[nnFrom.Grade]):
            Condition = Condition * False
            if ShowProcess == True:
                print('. False : Grade')
        if not (nnFrom.Thick <= ppTo.Thick and ppTo.Thick <= nnFrom.Thick + 2.):
            Condition = Condition * False
            if ShowProcess == True:
                print('. False : Thick')
        if not (ppTo.ProcessingStartDate == []):
            Date1 = (ppTo.ProcessingStartDate - nnFrom.ProcessingStartDate).days
            Date2 = (nnFrom.ProcessingStartDate - ppTo.ProcessingStartDate).days
            if not (Date1 <= 0 and Date2 <= 8):
                Condition = Condition * False
                if ShowProcess == True:
                    print('. False : Processing Start Date')
        return Condition


class Plate():
    ID = int(0)
    RollMarginX = 3.  # 롤마진 좌/우 각각 3.0 mm
    RollMarginY = 4.  # 롤마진 상/하 각각 4.0 mm
    plateXmin = 3000.  # 발주 가능 최소 길이
    plateYmin = 1000.  # 발주 가능 최소 폭
    plateXmax = 21000.  # 가공 가능 최대 길이 - 작업장 마다 다를 수 있으므로 추후 업데이트 필요함
    plateYmax = 4500.  # 가공 가능 최대 폭   - 작업장 마다 다를 수 있으므로 추후 업데이트 필요함

    def __init__(pp, PlateName, Grade, Thick, Xmax, Ymax, DateFrom, DateTo, ProcessingStage, isFixedSize):
        pp.ID = Plate.ID
        pp.PlateName = PlateName
        pp.Grade = Grade
        pp.Thick = float(Thick)
        pp.Xmin = 0.
        pp.Xmax = float(Xmax)
        pp.Ymin = 0.
        pp.Ymax = float(Ymax)
        pp.Area = (pp.Xmax - pp.Xmin) * (pp.Ymax - pp.Ymin)
        pp.DateFrom = datetime.strptime(DateFrom, '%Y%m%d').date()
        pp.DateTo = datetime.strptime(DateTo, '%Y%m%d').date()
        pp.ProcessingStage = ProcessingStage
        pp.isFixedSize = isFixedSize

        pp.RollMarginX = Plate.RollMarginX
        pp.RollMarginY = Plate.RollMarginY

        pp.DotXRollMargin = []
        pp.DotYRollMargin = []

        pp.PixelPlate = []
        pp.PixelPlateMargin = []

        pp.NodeID = []
        pp.PartID = []
        pp.ProcessingStartDate = []
        pp.ProcessingEndDate = []

        pp.XminExact = []
        pp.XmaxExact = []

        pp.YminExact = []
        pp.YmaxExact = []

        pp.LengthWithoutRollMargin = []
        pp.WidthWithoutRollMargin = []

        pp.LengthWithRollMargin = []
        pp.WidthWithRollMargin = []

        pp.ScrapRatio = []
        pp.Mass = []
        pp.SumPartMassInitial = []
        pp.SumPartMassFinal = []

        Plate.ID = Plate.ID + 1

    def RollMargin2Dot(pp, pixelSize):
        Xmin, Xmax = pp.Xmin, pp.Xmax
        Ymin, Ymax = pp.Ymin, pp.Ymax
        rollMarginX, rollMarginY = pp.RollMarginX, pp.RollMarginY

        X0, Y0 = Xmin + rollMarginX, Ymin + rollMarginY
        X1, Y1 = Xmax - rollMarginX, Ymin + rollMarginY
        X2, Y2 = Xmax - rollMarginX, Ymax - rollMarginY
        X3, Y3 = Xmin + rollMarginX, Ymax - rollMarginY
        Xs = [X0, X1, X2, X3, X0]
        Ys = [Y0, Y1, Y2, Y3, Y0]

        dotXRollMargin, dotYRollMargin = [], []
        for s in range(len(Xs) - 1):
            L = np.sqrt((Xs[s + 1] - Xs[s]) ** 2 + (Ys[s + 1] - Ys[s]) ** 2)
            noDot = int(np.floor(L / pixelSize)) + 5
            dotXSegment = np.linspace(Xs[s], Xs[s + 1], noDot)
            dotYSegment = np.linspace(Ys[s], Ys[s + 1], noDot)
            dotXRollMargin = np.hstack((dotXRollMargin, dotXSegment))
            dotYRollMargin = np.hstack((dotYRollMargin, dotYSegment))

        pp.DotXRollMargin = dotXRollMargin
        pp.DotYRollMargin = dotYRollMargin

    def Size2Dim(pp, pixelSize):
        dX = pp.Xmax - pp.Xmin
        dY = pp.Ymax - pp.Ymin

        if dY % pixelSize == 0.:
            dimI = int(np.floor(dY / pixelSize))
        else:
            dimI = int(np.floor(dY / pixelSize)) + 1

        if dX % pixelSize == 0.:
            dimJ = int(np.floor(dX / pixelSize))
        else:
            dimJ = int(np.floor(dX / pixelSize)) + 1

        return dimI, dimJ

    def Plate2Pixel(pp, pixelSize):

        dimI, dimJ = pp.Size2Dim(pixelSize)

        Xmin, Xmax = pp.Xmin, pp.Xmax
        Ymin, Ymax = pp.Ymin, pp.Ymax

        dotXRollMargin = pp.DotXRollMargin
        dotYRollMargin = pp.DotYRollMargin

        idxI = np.floor((Ymax - dotYRollMargin) / pixelSize)
        idxJ = np.floor((dotXRollMargin - Xmin) / pixelSize)
        idxI = idxI.astype(int)
        idxJ = idxJ.astype(int)

        pixelPlate = np.zeros((dimI, dimJ), dtype=int)
        pixelPlateMargin = np.zeros((dimI, dimJ), dtype=int)
        pixelPlateMargin[idxI, idxJ] = 2

        pp.PixelPlate = np.array(pixelPlate, dtype=int)
        pp.PixelPlateMargin = np.array(pixelPlateMargin, dtype=int)

    def CompilePlate(pp, pixelSize):
        pp.RollMargin2Dot(pixelSize)
        pp.Plate2Pixel(pixelSize)

    def CalculateScrapRatioAndWeight(pp, qList):

        plateArea = 0
        plateMass = 0

        steelDensityKgmm3 = 7.85e-6  # steel density kg/mm3

        print(
            '==================================================================================================================')
        if pp.__class__.__name__ == 'NewPlate':

            print('pp.ID=', pp.ID, ': New Plate')
            if pp.XminExact == []: pp.MeasurePlateSize(qList)

            plateArea = pp.LengthWithRollMargin * pp.WidthWithRollMargin  # mm2
            plateMass = steelDensityKgmm3 * plateArea * pp.Thick / 1000  # ton

        elif pp.__class__.__name__ == 'Plate':

            print('pp.ID=', pp.ID, ': Plate')
            plateArea = (pp.Xmax - pp.Xmin) * (pp.Ymax - pp.Ymin)  # mm2
            plateMass = steelDensityKgmm3 * plateArea * pp.Thick / 1000  # ton

        sumPartArea = 0
        sumPartMassInitial = 0
        sumPartMassFinal = 0

        if len(pp.PartID) == 0:
            print('* pp.PartID=', pp.PartID, ': no part allocated')
        else:
            print('* pp.PartID=', pp.PartID)
            # print('------------------------------------------------------------------------------------------------------------------')

            for qqID in pp.PartID:
                for qq in qList:
                    if qq.ID == qqID:
                        partArea = qq.Area  # mm2
                        partMassInitial = steelDensityKgmm3 * partArea * qq.Thick / 1000  # ton
                        partMassFinal = steelDensityKgmm3 * partArea * pp.Thick / 1000  # ton

                        sumPartArea = sumPartArea + partArea
                        sumPartMassInitial = sumPartMassInitial + partMassInitial
                        sumPartMassFinal = sumPartMassFinal + partMassFinal

                        # print('. qq.ID=',qq.ID,'partArea(mm^2)=',partArea, 'partMassInitial(ton)=', partMassInitial,'partMassFinal(ton)=',partMassFinal)

        scrapRatio = 1 - (sumPartArea / plateArea)

        print(
            '------------------------------------------------------------------------------------------------------------------')
        print('. plateMass =', plateMass, '(ton)')
        print('. sumPartMassInitial =', sumPartMassInitial, '(ton)')
        print('. sumPartMassFinal =', sumPartMassFinal, '(ton)')
        print(
            '..................................................................................................................')
        print('. scrapRatio =', scrapRatio)
        print('. sumPartMassInitial / plateMass =', sumPartMassInitial / plateMass)
        print('. sumPartMassFinal   / plateMass =', sumPartMassFinal / plateMass)

        pp.ScrapRatio = scrapRatio
        pp.Mass = plateMass
        pp.SumPartMassInitial = sumPartMassInitial
        pp.SumPartMassFinal = sumPartMassFinal

    def PlotPixel(pp, figSizeValue=12):
        # pixel 이미지 그리기
        plt.figure(figsize=(figSizeValue, figSizeValue))
        plt.imshow(np.array(pp.PixelPlate, dtype=int))

    def MeasurePlateSize(pp, qList):
        # 부재의 Dot 정보와 배치 정보를 이용해서 배치된 모든 부재의 외접사각형 사이즈 측정 후 자재 크기 업데이트
        qqIDList = pp.PartID

        XList, YList = [], []
        for qqID in qqIDList:
            for qqTemp in qList:
                if qqTemp.ID == qqID:
                    qq = qqTemp

                    Umid = (min(qq.DotUBurning) + max(qq.DotUBurning)) / 2
                    Vmid = (min(qq.DotVBurning) + max(qq.DotVBurning)) / 2

                    gdotU = qq.GDotUMargin
                    gdotV = qq.GDotVMargin
                    for idxG in range(len(gdotU)):
                        cdotU = gdotU[idxG] - Umid
                        cdotV = gdotV[idxG] - Vmid
                        rdotU, rdotV = qq.DotRotating(cdotU, cdotV, qq.AngleDegList[qq.AllocationIdxA[0]])
                        pdotU = qq.AllocationX[0] + rdotU
                        pdotV = qq.AllocationY[0] + rdotV
                        XList = XList + pdotU.tolist()
                        YList = YList + pdotV.tolist()

        XminExact = np.floor(min(XList))
        XmaxExact = np.ceil(max(XList))
        YminExact = np.floor(min(YList))
        YmaxExact = np.ceil(max(YList))

        lengthWithoutRollMargin = XmaxExact - XminExact
        widthWithoutRollMargin = YmaxExact - YminExact

        lengthWithRollMargin = lengthWithoutRollMargin + 2 * Plate.RollMarginX
        widthWithRollMargin = widthWithoutRollMargin + 2 * Plate.RollMarginY

        if lengthWithRollMargin < Plate.plateXmin:
            XmaxExact = XminExact + Plate.plateXmin - 2 * Plate.RollMarginX
            lengthWithoutRollMargin = XmaxExact - XminExact
            lengthWithRollMargin = lengthWithoutRollMargin + 2 * Plate.RollMarginX

        if widthWithRollMargin < Plate.plateYmin:
            YmaxExact = YminExact + Plate.plateYmin - 2 * Plate.RollMarginY
            widthWithoutRollMargin = YmaxExact - YminExact
            widthWithRollMargin = widthWithoutRollMargin + 2 * Plate.RollMarginY

        pp.XminExact = XminExact
        pp.XmaxExact = XmaxExact

        pp.YminExact = YminExact
        pp.YmaxExact = YmaxExact

        pp.LengthWithoutRollMargin = lengthWithoutRollMargin
        pp.LengthWithRollMargin = lengthWithRollMargin

        pp.WidthWithoutRollMargin = widthWithoutRollMargin
        pp.WidthWithRollMargin = widthWithRollMargin

    def PlotLine(pp, qList):
        # 자재 이미지 그리기

        qqIDList = pp.PartID

        if pp.__class__.__name__ == 'NewPlate':
            if pp.XminExact == []: pp.MeasurePlateSize(qList)
            plateXmin, plateXmax = pp.XminExact - Plate.RollMarginX, pp.XmaxExact + Plate.RollMarginX
            plateYmin, plateYmax = pp.YminExact - Plate.RollMarginY, pp.YmaxExact + Plate.RollMarginY
            rollMarginXmin, rollMarginXmax = plateXmin + Plate.RollMarginX, plateXmax - Plate.RollMarginX
            rollMarginYmin, rollMarginYmax = plateYmin + Plate.RollMarginY, plateYmax - Plate.RollMarginY

        elif pp.__class__.__name__ == 'Plate':
            plateXmin, plateXmax = pp.Xmin, pp.Xmax
            plateYmin, plateYmax = pp.Ymin, pp.Ymax
            rollMarginXmin, rollMarginXmax = plateXmin + Plate.RollMarginX, plateXmax - Plate.RollMarginX
            rollMarginYmin, rollMarginYmax = plateYmin + Plate.RollMarginY, plateYmax - Plate.RollMarginY

        plateXLine = [plateXmin, plateXmax, plateXmax, plateXmin, plateXmin]
        plateYLine = [plateYmin, plateYmin, plateYmax, plateYmax, plateYmin]

        rollMarginXLine = [rollMarginXmin, rollMarginXmax, rollMarginXmax, rollMarginXmin, rollMarginXmin]
        rollMarginYLine = [rollMarginYmin, rollMarginYmin, rollMarginYmax, rollMarginYmax, rollMarginYmin]

        plt.figure(figsize=((plateXmax - plateXmin) / 1000, (plateYmax - plateYmin) / 1000))
        plt.plot(plateXLine, plateYLine, color='k', linewidth='2')
        plt.plot(rollMarginXLine, rollMarginYLine, color='k', linewidth='1')

        for qqID in qqIDList:
            for qqTemp in qList:
                if qqTemp.ID == qqID:
                    qq = qqTemp

                    Umid = (min(qq.DotUBurning) + max(qq.DotUBurning)) / 2
                    Vmid = (min(qq.DotVBurning) + max(qq.DotVBurning)) / 2

                    gdotU = qq.GDotUBurning
                    gdotV = qq.GDotVBurning
                    for idxG in range(len(gdotU)):
                        cdotU = gdotU[idxG] - Umid
                        cdotV = gdotV[idxG] - Vmid
                        rdotU, rdotV = qq.DotRotating(cdotU, cdotV, qq.AngleDegList[qq.AllocationIdxA[0]])
                        pdotU = qq.AllocationX[0] + rdotU
                        pdotV = qq.AllocationY[0] + rdotV
                        if qq.InitialNodeID == qq.FinalNodeID:
                            plt.plot(pdotU, pdotV, color='r', linewidth='2')
                        else:
                            plt.plot(pdotU, pdotV, color='b', linewidth='2')

                    gdotU = qq.GDotUMarking
                    gdotV = qq.GDotVMarking
                    for idxG in range(len(gdotU)):
                        cdotU = gdotU[idxG] - Umid
                        cdotV = gdotV[idxG] - Vmid
                        rdotU, rdotV = qq.DotRotating(cdotU, cdotV, qq.AngleDegList[qq.AllocationIdxA[0]])
                        pdotU = qq.AllocationX[0] + rdotU
                        pdotV = qq.AllocationY[0] + rdotV
                        if qq.InitialNodeID == qq.FinalNodeID:
                            plt.plot(pdotU, pdotV, color='r', linewidth='1')
                        else:
                            plt.plot(pdotU, pdotV, color='b', linewidth='1')

                    gdotU = qq.GDotUMargin
                    gdotV = qq.GDotVMargin
                    for idxG in range(len(gdotU)):
                        cdotU = gdotU[idxG] - Umid
                        cdotV = gdotV[idxG] - Vmid
                        rdotU, rdotV = qq.DotRotating(cdotU, cdotV, qq.AngleDegList[qq.AllocationIdxA[0]])
                        pdotU = qq.AllocationX[0] + rdotU
                        pdotV = qq.AllocationY[0] + rdotV
                        if qq.InitialNodeID == qq.FinalNodeID:
                            plt.plot(pdotU, pdotV, color='r', linewidth='1')
                        else:
                            plt.plot(pdotU, pdotV, color='b', linewidth='1')


class NewPlate(Plate):
    ID = int(10000)

    def __init__(ppNew, nnID, nList, Xmax, Ymax, PlateName, pixelSize):

        for nnTemp in nList:
            if nnTemp.ID == nnID:
                nn = nnTemp

        if Xmax < Plate.plateXmin: Xmax = Plate.plateXmin  # 발주 가능 최소 길이 보다 작으면 발주 가능 최소 길이로 수정
        if Xmax > Plate.plateXmax: Xmax = Plate.plateXmax  # 가공 가능 최대 길이 보다   크면 가공 가능 최대 길이로 수정
        if Ymax < Plate.plateYmin: Ymax = Plate.plateYmin  # 발주 가능 최소   폭 보다 작으면 발주 가능 최소 폭으로 수정
        if Ymax > Plate.plateYmax: Ymax = Plate.plateYmax  # 가공 가능 최대   폭 보다   크면 가공 가능 최대 폭으로 수정

        ppNew.ID = NewPlate.ID
        ppNew.PlateName = PlateName
        ppNew.Grade = nn.Grade
        ppNew.Thick = nn.Thick
        ppNew.Xmin = 0.
        ppNew.Xmax = Xmax
        ppNew.Ymin = 0.
        ppNew.Ymax = Ymax
        ppNew.Area = (ppNew.Xmax - ppNew.Xmin) * (ppNew.Ymax - ppNew.Ymin)
        ppNew.DateFrom = nn.ProcessingStartDate  # 입고일은 가공착수일과 동일하다고 가정
        ppNew.DateTo = nn.ProcessingStartDate + timedelta(days=+40)  # 90일 보관 가정 (사내 40일, 사외 90일)
        ppNew.ProcessingStage = nn.ProcessingStage
        ppNew.isFixedSize = False

        ppNew.RollMarginX = Plate.RollMarginX
        ppNew.RollMarginY = Plate.RollMarginY

        ppNew.PixelPlate = []
        ppNew.PixelPlateMargin = []

        ppNew.NodeID = []
        ppNew.PartID = []

        ppNew.ProcessingStartDate = nn.ProcessingStartDate
        ppNew.ProcessingEndDate = nn.ProcessingEndDate

        ppNew.OX = []

        ppNew.XminExact = []
        ppNew.XmaxExact = []
        ppNew.YminExact = []
        ppNew.YmaxExact = []

        ppNew.ScrapRatio = []
        ppNew.Mass = []
        ppNew.SumPartMassInitial = []
        ppNew.SumPartMassFinal = []

        Plate.CompilePlate(ppNew, pixelSize)

        NewPlate.ID = NewPlate.ID + 1


def Read_ScheduleFile(ScheduleFilePath,ScheduleFileName):
    # 스케쥴 파일 읽어오는 함수
    ScheduleFile  = open(ScheduleFilePath+'\\'+ScheduleFileName,'r')
    ScheduleFile_Lines = csv.reader(ScheduleFile)
    sList,lineNumber = [],0
    for ScheduleFile_Line in ScheduleFile_Lines:
        if not(lineNumber == 0):
            Dock                   = ScheduleFile_Line[ 0]  # 도크
            ShipNo                 = ScheduleFile_Line[ 1]  # 호선
            LOT                    = ScheduleFile_Line[ 2]  # 로트
            ProcessingBay          = ScheduleFile_Line[ 4]  # 가공BAY(ex.사외)
            ProcessingStage        = ScheduleFile_Line[ 5]  # 가공STAGE1
            ProcessingStartDate    = ScheduleFile_Line[ 7]  # 가공착수일(yyyymmdd)
            ProcessingEndDate      = ScheduleFile_Line[ 8]  # 가공완료일(yyyymmdd)
            ShipType               = ScheduleFile_Line[22]  # 선종(ex.300k VLCC)
            ss = Schedule(Dock,ShipNo,LOT,ProcessingBay,ProcessingStage,ProcessingStartDate,ProcessingEndDate,ShipType)
            sList = sList+[ss]
        lineNumber = lineNumber + 1
    ScheduleFile.close()
    return sList


def Read_PartListFile(PartListFilePath,PartListFileName):
    # 부재 리스트 파일 읽어오는 함수
    PartListFile  = open(PartListFilePath+'\\'+PartListFileName,'r')
    PartListFile_Lines = csv.reader(PartListFile)
    qList,lineNumber = [],0
    for PartListFile_Line in PartListFile_Lines:
        if not(lineNumber == 0):
            ShipNo       = PartListFile_Line[ 0] # 호선
            Block        = PartListFile_Line[ 1] # 블록
            TribonName   = PartListFile_Line[ 5] # 부재명
            Type         = PartListFile_Line[ 6] # 부재타입: angle, bracket, cutout, doubling plate, pillar, plate, standard, stiffener
            Symmetry     = PartListFile_Line[ 7] #
            P            = PartListFile_Line[ 8] #
            S            = PartListFile_Line[ 9] #
            Grade        = PartListFile_Line[10] # 재질
            Area         = PartListFile_Line[12] # 면적
            Thick        = PartListFile_Line[13] # 두께
            LOT          = PartListFile_Line[19] # 로트
            WorkingStage = PartListFile_Line[20] #
            FAB          = PartListFile_Line[30] # 가공계열
            # angle과 pillar는 네스팅 설계의 대상이 아니므로 제외.
            # 부재리스트(엑셀파일) 중 두께가 0. 또는 면적이 0.인 경우도 제외.
            #if not(Type == 'angle' or Type == 'pillar' or Type == 'stiffener'):
            if not(Type == 'angle' or Type == 'pillar'):
                if not('/' in Thick):
                    if not(float(Thick)*float(Area)==0):
                        qq = Part(ShipNo,Block,TribonName,Type,Symmetry,P,S,Grade,Area,Thick,LOT,WorkingStage,FAB)
                        qList = qList + [qq]
        lineNumber = lineNumber + 1
    PartListFile.close()
    return qList


def ListUp_GenFileName(GenFilePath,GenFileExtn):
    # 부재 gen 파일 리스트업하는 함수
    AllFileNameList = os.listdir(GenFilePath)
    GenFileNameList = []
    for FileName in AllFileNameList:
        if FileName[-4:] == GenFileExtn:
            GenFileNameList = GenFileNameList + [FileName]
    return GenFileNameList


def Read_GradeMixFile(GradeMixFilePath,GradeMixFileName):
    # 재질합 규칙 파일 읽어오는 함수
    GradeMixFile = open(GradeMixFilePath+'\\'+GradeMixFileName,'r')
    GradeMixFile_Lines = csv.reader(GradeMixFile)
    gList,lineNumber = [],0
    for GradeMixFile_Line in GradeMixFile_Lines:
        if not(lineNumber == 0):
            ggFrom = GradeMixFile_Line[0]
            ggTo   = [ i for i in GradeMixFile_Line[1:] if i not in [''] ]
            ggTemp = [ggFrom, ggTo]
            isThereSameGrade=0 #0:No, 1:Yes
            for gg in gList:
                if ggTemp[0] == gg[0]:
                    gg[1] = gg[1]+ggTemp[1]
                    isThereSameGrade = 1
            if isThereSameGrade == 0:
                gList = gList+[ggTemp]
        lineNumber = lineNumber + 1
    GradeMixFile.close()
    gDict = {}
    for gg in gList:
        gDict.update({gg[0]:set(gg[1])})
    return gDict


def Read_PlateListFile(PlateListFilePath, PlateListFileName):
    # 기존 잔재 및 사이즈 확정 자재 리스트 파일 읽어오는 함수
    PlateListFile = open(PlateListFilePath + '\\' + PlateListFileName, 'r')
    PlateListFile_Lines = csv.reader(PlateListFile)
    pList, lineNumber = [], 0
    for PlateListFile_Line in PlateListFile_Lines:
        if not (lineNumber == 0):
            PlateName = PlateListFile_Line[0]
            Grade = PlateListFile_Line[1]
            Thick = PlateListFile_Line[2]
            Xmax = PlateListFile_Line[3]
            Ymax = PlateListFile_Line[4]
            DateFrom = PlateListFile_Line[5]
            DateTo = PlateListFile_Line[6]
            ProcessingStage = PlateListFile_Line[7]
            isFixedSize = True
            pp = Plate(PlateName, Grade, Thick, Xmax, Ymax, DateFrom, DateTo, ProcessingStage, isFixedSize)
            pList = pList + [pp]
        lineNumber = lineNumber + 1
    PlateListFile.close()
    return pList


def Get_DGmat(nList, ShowProcess=False, figSizeValue=8):
    # DG 행렬 생성하는 함수
    DG = np.zeros((len(nList), len(nList)), dtype=int)
    DAG = np.zeros((len(nList), len(nList)), dtype=int)
    for nnFrom in nList:
        for nnTo in nList:
            if nnFrom.IsPossibleToGoToNode(nnTo, gDict):
                DG[nnFrom.ID, nnTo.ID] = int(1)
                if nnFrom.ID != nnTo.ID:
                    DAG[nnFrom.ID, nnTo.ID] = int(1)

    if ShowProcess == True:
        print('DG matrix shape:', DG.shape)
        plt.figure(figsize=(figSizeValue, figSizeValue))
        plt.imshow(DG)
        plt.show()

    return DG, DAG


def Select_nn(endNodeIDList, nList, ShowProcess=False):
    # 여러 개의 말단노드 중 다음의 정렬 기준에 따라 하나를 선정
    # (1)Grade --> (2)Thickness --> (3)ProcessingStartDate --> (4)Select first one

    endNodeList = []
    for endNodeID in endNodeIDList:
        for nn in nList:
            if nn.ID == endNodeID:
                endNodeList = endNodeList + [nn]

    # Sorting1: Grade
    endNodeList_New = []
    for endNodeFrom in endNodeList:
        isFinalGrade = True
        for endNodeTo in endNodeList:
            if endNodeFrom.ID != endNodeTo.ID:
                if endNodeTo.Grade in gDict[endNodeFrom.Grade]:
                    isFinalGrade = False
        if isFinalGrade == True:
            endNodeList_New = endNodeList_New + [endNodeFrom]
    # 참고: 모든 말단 노드의 Grade가 동일한 경우 예외 처리
    if len(endNodeList_New) == 0:
        endNodeList_New = copy.deepcopy(endNodeList)
    endNodeList = copy.deepcopy(endNodeList_New)
    if ShowProcess == True:
        print('===================================')
        print('Select_nn - step 1: Grade')
        print('-----------------------------------')
        for endNode in endNodeList: print(endNode.ID, endNode.Grade, endNode.Thick, endNode.ProcessingStartDate)

    # Sorting2: Thick
    ThickMax = 0.
    endNodeList_New = []
    for endNode in endNodeList:
        if ThickMax < endNode.Thick:
            ThickMax = endNode.Thick
            endNodeList_New = [endNode]
        elif ThickMax == endNode.Thick:
            endNodeList_New = endNodeList_New + [endNode]
    endNodeList = copy.deepcopy(endNodeList_New)
    if ShowProcess == True:
        print('===================================')
        print('Select_nn - step 2: Thick')
        print('-----------------------------------')
        for endNode in endNodeList: print(endNode.ID, endNode.Grade, endNode.Thick, endNode.ProcessingStartDate)

    # Sorting3: ProcessingStartDate
    ProcessingStartDateEarliest = datetime.strptime('99991231', '%Y%m%d').date()
    endNodeList_New = []
    for endNode in endNodeList:
        if ProcessingStartDateEarliest > endNode.ProcessingStartDate:
            ProcessingStartDateEarliest = endNode.ProcessingStartDate
            endNodeList_New = [endNode]
        elif ProcessingStartDateEarliest == endNode.ProcessingStartDate:
            endNodeList_New = endNodeList_New + [endNode]
    endNodeList = copy.deepcopy(endNodeList_New)
    if ShowProcess == True:
        print('===================================')
        print('Select_nn - step 3: ProcessingStartDate')
        print('-----------------------------------')
        for endNode in endNodeList: print(endNode.ID, endNode.Grade, endNode.Thick, endNode.ProcessingStartDate)

    # Sorting4: Select first one
    endNode = endNodeList[0]
    endNodeID = endNode.ID
    if ShowProcess == True:
        print('===================================')
        print('Select_nn - step 4: Select first one')
        print('-----------------------------------')
        print(endNode.ID, endNode.Grade, endNode.Thick, endNode.ProcessingStartDate)
        print('===================================')

    return endNodeID


def Select_pp(pList, nnID, nList, ShowProcess=False):
    # 확정 자재 또는 잔재 선택하기

    # 말단 노드 객체 가져오기
    for nn in nList:
        if nn.ID == nnID:
            nnFrom = nn

    ppToList = pList

    # Sorting A: 사용 이력이 없는 자재만 선택하기
    ppToList_New = []
    for ppTo in ppToList:
        if ppTo.PartID == []:
            ppToList_New = ppToList_New + [ppTo]
    ppToList = copy.deepcopy(ppToList_New)
    if ShowProcess == True:
        print('===================================')
        print('Select_pp - step A: Select not-used plate')
        print('-----------------------------------')
        for ppTo in ppToList: print(ppTo.ID, ppTo.PartID, ppTo.Grade, ppTo.Thick, ppTo.DateTo, ppTo.Area,
                                    ppTo.ProcessingStartDate)

    # Sorting 0: 합네스팅(nnFrom --> ppTo) 가능 자재 분류하기
    ppToList_New = []
    for ppTo in ppToList:
        if nnFrom.IsPossibleToGoToPlate(ppTo, gDict):
            ppToList_New = ppToList_New + [ppTo]
    ppToList = copy.deepcopy(ppToList_New)
    if ShowProcess == True:
        print('===================================')
        print('Select_pp - step 0: HapNesting Possible')
        print('-----------------------------------')
        for ppTo in ppToList: print(ppTo.ID, ppTo.PartID, ppTo.Grade, ppTo.Thick, ppTo.DateTo, ppTo.Area,
                                    ppTo.ProcessingStartDate)

    # Sorting 1: Grade
    ppToList_New = []
    for ppFrom in ppToList:
        isFinalGrade = True
        for ppTo in ppToList:
            if ppFrom.ID != ppTo.ID:
                if ppTo.Grade in gDict[ppFrom.Grade]:
                    isFinalGrade = False
        if isFinalGrade == True:
            ppToList_New = ppToList_New + [ppFrom]
    if len(ppToList_New) == 0:  # 참고: 모든 말단 노드의 재질이 동일한 경우 예외 처리
        ppToList_New = copy.deepcopy(ppToList)
    ppToList = copy.deepcopy(ppToList_New)
    if ShowProcess == True:
        print('===================================')
        print('Select_pp - step 1: Grade')
        print('-----------------------------------')
        for ppTo in ppToList: print(ppTo.ID, ppTo.PartID, ppTo.Grade, ppTo.Thick, ppTo.DateTo, ppTo.Area,
                                    ppTo.ProcessingStartDate)

    # Sorting2: Thick
    ThickMax = 0.
    ppToList_New = []
    for ppTo in ppToList:
        if ThickMax < ppTo.Thick:
            ThickMax = ppTo.Thick
            ppToList_New = [ppTo]
        elif ThickMax == ppTo.Thick:
            ppToList_New = ppToList_New + [ppTo]
    ppToList = copy.deepcopy(ppToList_New)
    if ShowProcess == True:
        print('===================================')
        print('Select_pp - step 2: Thick')
        print('-----------------------------------')
        for ppTo in ppToList: print(ppTo.ID, ppTo.PartID, ppTo.Grade, ppTo.Thick, ppTo.DateTo, ppTo.Area,
                                    ppTo.ProcessingStartDate)

    # Sorting3: 잔재 폐기일(pp.DateTo)
    DateToEarliest = datetime.strptime('99991231', '%Y%m%d').date()
    ppToList_New = []
    for ppTo in ppToList:
        if DateToEarliest > ppTo.DateTo:
            DateToEarliest = ppTo.DateTo
            ppToList_DateTo = [ppTo]
        elif DateToEarliest == ppTo.DateTo:
            ppToList_New = ppToList_New + [ppTo]
    ppToList = copy.deepcopy(ppToList_New)
    if ShowProcess == True:
        print('===================================')
        print('Select_pp - step 3: DateTo')
        print('-----------------------------------')
        for ppTo in ppToList: print(ppTo.ID, ppTo.PartID, ppTo.Grade, ppTo.Thick, ppTo.DateTo, ppTo.Area,
                                    ppTo.ProcessingStartDate)

    # Sorting4: 면적(pp.Area)
    AreaMin = 1.e+12
    ppToList_New = []
    for ppTo in ppToList:
        if AreaMin > ppTo.Area:
            AreaMin = ppTo.Area
            ppToList_New = [ppTo]
        elif AreaMin == ppTo.Area:
            ppToList_New = ppToList_New + [ppTo]
    ppToList = copy.deepcopy(ppToList_New)
    if ShowProcess == True:
        print('===================================')
        print('Select_pp - step 4: Area')
        print('-----------------------------------')
        for ppTo in ppToList: print(ppTo.ID, ppTo.PartID, ppTo.Grade, ppTo.Thick, ppTo.DateTo, ppTo.Area,
                                    ppTo.ProcessingStartDate)

    # Sorting5: First one
    ppToID = []
    if len(ppToList) == 0:
        ppToID = []
    else:
        ppTo = ppToList[0]
        ppToID = ppTo.ID
    if ShowProcess == True:
        print('===================================')
        print('Select_pp - step 5: First one')
        print('-----------------------------------')
        for ppTo in ppToList: print(ppTo.ID, ppTo.PartID, ppTo.Grade, ppTo.Thick, ppTo.DateTo, ppTo.Area,
                                    ppTo.ProcessingStartDate)

    # 플레이트 객체에 가공착수일 정보 업데이트
    for pp in pList:
        if pp.ID == ppToID:
            pp.ProcessingStartDate = nnFrom.ProcessingStartDate
            if ShowProcess == True:
                print('===================================')
                print('Select_pp - Plate processing start date updated')
                print('-----------------------------------')
                print('* Plate ID:', pp.ID)
                print('* Plate processing start date:', pp.ProcessingStartDate)
                print('===================================')

    return ppToID


def Get_Qn(nnID, nList, qList):
    # 잎서 선택한 말단 노드의 부재들의 ID 리스트(QnTempID)를 리턴

    for nn in nList:
        if nn.ID == nnID:
            QnID = [nn.Qn]

    QnID = Sort_QnID(QnID, qList)

    return QnID


def Get_Rn(ppID, pList, qList, nnID, nList):
    # 앞서 선택한 잔재 또는 말단 노드로 합네스팅 가능 부재들의 ID 리스트(RnTempID)를 리턴

    To_Grade = []
    To_Thick = []
    To_ProcessingStartDate = []
    if ppID == []:  # 합네스팅 할 수 있는 잔재가 없는 경우
        nnToID = nnID
        nnTo = []
        for nn in nList:
            if nn.ID == nnToID:
                nnTo = nn
        nnFromID = []
        for nnFromIDTemp in range(DG.shape[0]):
            if nnFromIDTemp != nnToID and DG[nnFromIDTemp, nnToID] == 1:
                nnFromID = nnFromID + [nnFromIDTemp]
        nnFrom = []
        for nnFromIDTemp in nnFromID:
            nnFromTemp = []
            for nn in nList:
                if nn.ID == nnFromIDTemp:
                    nnFromTemp = nn
            nnFrom = nnFrom + [nnFromTemp]
        To_Grade = nnTo.Grade
        To_Thick = nnTo.Thick
        To_ProcessingStartDate = nnTo.ProcessingStartDate
    else:
        ppToID = ppID
        ppTo = []
        for pp in pList:
            if pp.ID == ppToID:
                ppTo = pp
        nnFrom = []
        for nnFromTemp in nList:
            if nnFromTemp.ID != nnID and nnFromTemp.IsPossibleToGoToPlate(ppTo, gDict) == True:
                nnFrom = nnFrom + [nnFromTemp]
        To_Grade = ppTo.Grade
        To_Thick = ppTo.Thick
        To_ProcessingStartDate = ppTo.ProcessingStartDate
    nnFromOrderedID = [[], [], [], [], [], [], [], []]
    for nnFromTemp in nnFrom:
        isSameGrade = nnFromTemp.Grade == To_Grade
        isSameThick = nnFromTemp.Thick == To_Thick
        isSameProcessingStartDate = nnFromTemp.ProcessingStartDate == To_ProcessingStartDate
        if isSameGrade == True and isSameThick == True and isSameProcessingStartDate == True:  # 0: same Grade, Thick, ProcessingStartDate
            nnFromOrderedID[0].append(nnFromTemp.ID)
        if isSameGrade == False and isSameThick == True and isSameProcessingStartDate == True:  # 1: Grade mix
            nnFromOrderedID[1].append(nnFromTemp.ID)
        if isSameGrade == True and isSameThick == False and isSameProcessingStartDate == True:  # 2: Thick mix
            nnFromOrderedID[2].append(nnFromTemp.ID)
        if isSameGrade == False and isSameThick == False and isSameProcessingStartDate == True:  # 3: Grade+Thick mix
            nnFromOrderedID[3].append(nnFromTemp.ID)
        if isSameGrade == True and isSameThick == True and isSameProcessingStartDate == False:  # 4: ProcessingStartDate mix
            nnFromOrderedID[4].append(nnFromTemp.ID)
        if isSameGrade == False and isSameThick == True and isSameProcessingStartDate == False:  # 5: ProcessingStartDate+Grade mix
            nnFromOrderedID[5].append(nnFromTemp.ID)
        if isSameGrade == True and isSameThick == False and isSameProcessingStartDate == False:  # 6: ProcessingStartDate+Thick mix
            nnFromOrderedID[6].append(nnFromTemp.ID)
        if isSameGrade == False and isSameThick == False and isSameProcessingStartDate == False:  # 7: ProcessingStartDate+Grade+Thick mix
            nnFromOrderedID[7].append(nnFromTemp.ID)

    RnID = []
    for nnFromOrderedIDTemp in nnFromOrderedID:
        for nnFromOrderedIDTemp2 in nnFromOrderedIDTemp:
            for nn in nList:
                if nn.ID == nnFromOrderedIDTemp2:
                    RnID = RnID + [nn.Qn]

    RnID = Sort_QnID(RnID, qList)

    return RnID


def conv2d_ForLoop(X, kernels, stride=1, padding='same', padValue=1):
    # for-loop를 이용한 부재/자재 간 convolution 연산

    N_X, C_X, H_X, W_X = X.shape  # shape of input image (X)
    N_kernel, C_kernel, H_kernel, W_kernel = kernels.shape  # shape of kernels

    # shape of output image (Y)
    if padding == 'same': H_pad, W_pad = H_kernel // 2, W_kernel // 2
    if padding == 'valid': H_pad, W_pad = 0, 0
    H_Y = (H_X + 2 * H_pad - H_kernel) // stride + 1
    W_Y = (W_X + 2 * W_pad - W_kernel) // stride + 1

    # padding to height(H_X) and width(W_X) of image(X)
    X_padded = np.pad(X, [(0, 0), (0, 0), (H_pad, H_pad), (W_pad, W_pad)], 'constant', constant_values=padValue)
    X_padded = X_padded.astype('int')

    # convolution using For-Loop
    Y = np.zeros((N_X, N_kernel, H_Y, W_Y), dtype='int')
    for n in range(N_X):  # for each input image.
        for c in range(N_kernel):  # for each output channel.
            for h in range(H_Y):  # slide the filter vertically.
                hStart = h * stride
                hEnd = hStart + H_kernel
                for w in range(W_Y):  # slide the filter horizontally.
                    wStart = w * stride
                    wEnd = wStart + W_kernel
                    # Element-wise multiplication.
                    Y[n, c, h, w] = np.sum(X_padded[n, :, hStart:hEnd, wStart:wEnd] * kernels[c])
    return Y


def conv2d_Im2Col(X, kernels, stride=1, padding='same', padValue=1):
    # im2col을 이용한 부재/자재 간 convolution 연산

    N_X, C_X, H_X, W_X = X.shape  # shape of input image (X)
    N_kernel, C_kernel, H_kernel, W_kernel = kernels.shape  # shape of kernels

    # shape of output image (Y)
    if padding == 'same': H_pad, W_pad = H_kernel // 2, W_kernel // 2
    if padding == 'valid': H_pad, W_pad = 0, 0
    H_Y = (H_X + 2 * H_pad - H_kernel) // stride + 1
    W_Y = (W_X + 2 * W_pad - W_kernel) // stride + 1

    # padding to height(H_X) and width(W_X) of image(X)
    X_padded = np.pad(X, [(0, 0), (0, 0), (H_pad, H_pad), (W_pad, W_pad)], 'constant', constant_values=padValue)
    X_padded = X_padded.astype('int')

    # convolution using Vectorization
    # step1: im2col
    Xcol = np.zeros((N_X, C_X, H_kernel, W_kernel, H_Y, W_Y), dtype='int')
    for h in range(H_kernel):
        hEnd = h + stride * H_Y
        for w in range(W_kernel):
            wEnd = w + stride * W_Y
            Xcol[:, :, h, w, :, :] = X_padded[:, :, h:hEnd:stride, w:wEnd:stride]
    Xcol = Xcol.transpose(0, 4, 5, 1, 2, 3).reshape(N_X * H_Y * W_Y, -1)
    # step2: matrix multiplication
    Y = np.matmul(Xcol, kernels.reshape(N_kernel, -1).T)
    Y = Y.reshape(N_X, H_Y, W_Y, N_kernel)
    Y = Y.transpose(0, 3, 1, 2)

    return Y


def Update_PixelPlate(plate, qq, idxA, idxH, idxW, ShowProcess=False, figSizeValue=12):
    _, _, plateH, plateW = plate.shape

    ## part pixel image formatting
    partH, partW, partA = qq.PixelPart.shape

    part = np.zeros((partA, 1, partH, partW), dtype=int)
    for a in range(partA):
        part[a, 0, :, :] = np.array(qq.PixelPart[:, :, a], dtype=int) + np.array(qq.PixelMargin[:, :, a], dtype=int)

    idxH_pad = idxH[0] + partH // 2
    idxW_pad = idxW[0] + partW // 2

    plate_pad = np.pad(plate[0, 0, :, :], ((partW // 2, partW // 2), (partH // 2, partH // 2)), 'constant',
                       constant_values=0)
    plate_pad[idxH_pad - partH // 2:idxH_pad + partH // 2 + 1,
    idxW_pad - partW // 2:idxW_pad + partW // 2 + 1] = plate_pad[idxH_pad - partH // 2:idxH_pad + partH // 2 + 1,
                                                       idxW_pad - partW // 2:idxW_pad + partW // 2 + 1] + part[idxA[0],
                                                                                                          0, :, :]
    plate_updated = plate_pad[partH // 2:-partH // 2 + 1, partW // 2:-partW // 2 + 1]
    plate = np.array([[plate_updated]], dtype=int)

    if ShowProcess == True:
        plt.figure(figsize=(figSizeValue, figSizeValue))
        plt.imshow(plate_updated)

    return plate


def Sort_QnID(QnID, qList):
    # Rn Sorting: Area
    for r, qnID in enumerate(QnID):
        qqIDList = []
        areaList = []
        for qqID in qnID:
            for qq in qList:
                if qq.ID == qqID:
                    qqIDList = qqIDList + [qq.ID]
                    areaList = areaList + [qq.Area]
        qqIDList = np.array(qqIDList)
        areaList = np.array(areaList)
        qqIDList = qqIDList[np.argsort(areaList)[::-1]]
        qqIDList = qqIDList.tolist()
        QnID[r] = qqIDList

    return QnID


# DG 행렬로부터 말단 노드 ID 추출
def Get_endNodeList(DG):
    numberOfNode, _ = DG.shape

    minNumberOfNodeTo = numberOfNode
    for n in range(numberOfNode):
        if 0 < np.sum(DG[n, :]) and np.sum(DG[n, :]) < minNumberOfNodeTo:
            minNumberOfNodeTo = np.sum(DG[n, :])

    endNodeIDList = []
    for n in range(numberOfNode):
        if np.sum(DG[n, :]) == minNumberOfNodeTo:
            endNodeIDList = endNodeIDList + [n]

    return endNodeIDList


def Update_DGmat(DG, DAG, nnID, ShowProcess=False, figSizeValue=8):
    DG[nnID, :] = 0
    DG[:, nnID] = 0

    DAG[nnID, :] = 0
    DAG[:, nnID] = 0

    if ShowProcess == True:
        print('DG matrix shape:', DG.shape)
        plt.figure(figsize=(figSizeValue, figSizeValue))
        plt.imshow(DG)
        plt.show()

    return DG, DAG


def P2QR(ppID, pList, QnTempID, qList, nnID, nList, ShowProcess=False):
    # plate pixel image formatting
    for ppTemp in pList:
        if ppTemp.ID == ppID:
            pp = ppTemp

    plate = np.array([[pp.PixelPlate]], dtype=int)
    plateMargin = np.array([[pp.PixelPlateMargin]], dtype=int)

    minX = Plate.plateXmax
    maxX = 0.
    minY = Plate.plateYmax
    maxY = 0.

    # Allocation using multi-layered objective function
    for qnTempID in QnTempID:
        for qqID in qnTempID:

            for qqTemp in qList:
                if qqTemp.ID == qqID: qq = qqTemp

            idxA, idxH, idxW, allocationA, allocationY, allocationX, minXTemp, maxXTemp, minYTemp, maxYTemp = Argmin_Loss(
                plateMargin, qq, minX, maxX, minY, maxY)  # ,ShowProcess=True)

            ### Update the allocation results
            if idxA != []:
                if minX > minXTemp: minX = minXTemp
                if maxX < maxXTemp: maxX = maxXTemp
                if minY > minYTemp: minY = minYTemp
                if maxY < maxYTemp: maxY = maxYTemp

                plateMargin = Update_PixelPlate(plateMargin, qq, idxA, idxH, idxW)  # ,ShowProcess=True)
                plate = Update_PixelPlate(plate, qq, idxA, idxH, idxW)  # ,ShowProcess=True)

                qq.AllocationX = [allocationX]
                qq.AllocationY = [allocationY]
                qq.AllocationA = [allocationA]

                qq.AllocationIdxW = idxW
                qq.AllocationIdxH = idxH
                qq.AllocationIdxA = idxA

                qq.PlateID = ppID
                qq.FinalNodeID = nnID

                pp.PixelPlateMargin = np.array(plateMargin[0, 0, :, :], dtype='int')
                pp.PixelPlate = np.array(plate[0, 0, :, :], dtype='int')

                pp.PartID.append(qqID)
                pp.NodeID.append(qq.InitialNodeID)

                initialNodeID = qq.InitialNodeID
                for nnTemp in nList:
                    if nnTemp.ID == initialNodeID:
                        nn = nnTemp
                        nn.AllocatedQn.append(qqID)
                        nn.Qn.remove(qqID)

                if ShowProcess == True:
                    print('---------------------------------------')
                    print('qq.ID:', qq.ID)
                    print('GenFileName:', qq.GenFileName)
                    print('idxA=', idxA, 'idxH=', idxH, 'idxW=', idxW)
                    qq.PlotPixel()


def Q2P(pList, QnTempID, qList, nnID, nList, pixelSize, finalResolution=2000, ShowProcess=False):
    plateXmin = Plate.plateXmin
    plateYmin = Plate.plateYmin
    plateXmax = Plate.plateXmax
    plateYmax = Plate.plateYmax

    ## Node Provision
    nList_Q2P_Original = []
    for nnTemp in nList:
        if nnTemp.ID == nnID:
            nList_Q2P_Original = nList_Q2P_Original + [nnTemp]
    nList_Q2P = copy.deepcopy(nList_Q2P_Original)

    ## Part Provision
    qList_Q2P_Original = []
    for qnTempID in QnTempID:
        for qqTempID in qnTempID:
            for qqTemp in qList:
                if qqTemp.ID == qqTempID:
                    qList_Q2P_Original = qList_Q2P_Original + [qqTemp]
    qList_Q2P = copy.deepcopy(qList_Q2P_Original)

    totalNetArea_Qn = 0.
    for qq in qList_Q2P:
        totalNetArea_Qn = totalNetArea_Qn + qq.Area

    ## 자재 사이즈 결정 1단계: 2차원 이진탐색으로 대략적인 사이즈 찾기
    pList_Q2P = []
    temporary_pp0 = NewPlate(nnID, nList, plateXmax, plateYmax, 'TemporaryPlate', pixelSize)
    pList_Q2P = pList_Q2P + [temporary_pp0]

    pList_Q2P_Record = [[temporary_pp0]]

    P2QR(temporary_pp0.ID, pList_Q2P, QnTempID, qList_Q2P, nnID, nList_Q2P)  # , ShowProcess=True)
    if len(temporary_pp0.PartID) == len(qList_Q2P):
        temporary_pp0.OX = True
    else:
        temporary_pp0.OX = False

    print(' . temporary_pp0.ID=', temporary_pp0.ID, 'Xmax=', temporary_pp0.Xmax, 'Ymax=', temporary_pp0.Ymax, '--> OX=',
          temporary_pp0.OX)
    temporary_pp0.PlotPixel()

    if temporary_pp0.OX == True:  # 최대 사이즈 자재에 QnTempID의 모든 부재가 배치된 경우
        ppLayerNow = [temporary_pp0]
        NumOfLayer = 0
        dX, dY = plateXmax / 2, plateYmax / 2

        NumOfLayer = NumOfLayer + 1
        while max(dX, dY) >= finalResolution:

            ppLayerNew = []

            for pp in ppLayerNow:  # 2**NumOfLayer 만큼 반복하는데, 멀티스레드 적용 가능할 듯.
                if pp.OX == True:
                    X1, Y1 = pp.Xmax - dX, pp.Ymax  # 사이즈 줄이기 - 길이 방향으로
                    X2, Y2 = pp.Xmax, pp.Ymax - dY  # 사이즈 줄이기 -   폭 방향으로
                else:
                    X1, Y1 = pp.Xmax + dX, pp.Ymax  # 사이즈 늘리기 - 길이 방향으로
                    X2, Y2 = pp.Xmax, pp.Ymax + dY  # 사이즈 늘리기 -   폭 방향으로

                # 사이즈 줄이기 또는 늘리기 - 길이 방향으로
                temporary_pp1 = NewPlate(nnID, nList, X1, Y1, 'TemporaryPlate', pixelSize)
                pList_Q2P = pList_Q2P + [temporary_pp1]
                qList_Q2P = copy.deepcopy(qList_Q2P_Original)
                nList_Q2P = copy.deepcopy(nList_Q2P_Original)
                if totalNetArea_Qn < temporary_pp1.Area:
                    P2QR(temporary_pp1.ID, pList_Q2P, QnTempID, qList_Q2P, nnID, nList_Q2P)  # , ShowProcess=True)
                    if len(temporary_pp1.PartID) == len(qList_Q2P):
                        temporary_pp1.OX = True
                    else:
                        temporary_pp1.OX = False
                else:
                    temporary_pp1.OX = False
                ppLayerNew = ppLayerNew + [temporary_pp1]
                pList_Q2P_Record = pList_Q2P_Record + [[temporary_pp1]]
                print(' . temporary_pp1.ID=', temporary_pp1.ID, 'Xmax=', temporary_pp1.Xmax, 'Ymax=',
                      temporary_pp1.Ymax, '--> OX=', temporary_pp1.OX)
                temporary_pp1.PlotPixel()

                # 사이즈 줄이기 또는 늘리기 -   폭 방향으로
                temporary_pp2 = NewPlate(nnID, nList, X2, Y2, 'TemporaryPlate', pixelSize)
                pList_Q2P = pList_Q2P + [temporary_pp2]
                qList_Q2P = copy.deepcopy(qList_Q2P_Original)
                nList_Q2P = copy.deepcopy(nList_Q2P_Original)
                if totalNetArea_Qn < temporary_pp2.Area:
                    P2QR(temporary_pp2.ID, pList_Q2P, QnTempID, qList_Q2P, nnID, nList_Q2P)  # , ShowProcess=True)
                    if len(temporary_pp2.PartID) == len(qList_Q2P):
                        temporary_pp2.OX = True
                    else:
                        temporary_pp2.OX = False
                else:
                    temporary_pp2.OX = False
                ppLayerNew = ppLayerNew + [temporary_pp2]
                pList_Q2P_Record = pList_Q2P_Record + [[temporary_pp2]]
                print(' . temporary_pp2.ID=', temporary_pp2.ID, 'Xmax=', temporary_pp2.Xmax, 'Ymax=',
                      temporary_pp2.Ymax, '--> OX=', temporary_pp2.OX)
                temporary_pp2.PlotPixel()

            ppLayerNow = copy.deepcopy(ppLayerNew)
            dX, dY = dX / 2, dY / 2
            NumOfLayer = NumOfLayer + 1

    ## 자재 사이즈 결정 2-1단계: 부재의 배치 정보로 자재 사이즈 맞춤하기 - pixel 이미지 용
    netXmaxList, netYmaxList, netAreaList = [], [], []
    netDimJList, netDimIList = [], []
    if temporary_pp0.OX == True:
        for ppTemp in pList_Q2P_Record:
            pp = ppTemp[0]
            if pp.OX == True:
                pixelPlate = pp.PixelPlate
                idxI, idxJ = np.where(pixelPlate > 0)
                netDimI = max(idxI) - min(idxI) + 1
                netDimJ = max(idxJ) - min(idxJ) + 1

                # 발주 가능 최소 길이/폭 반영 위해 코드 추가(여기부터)
                minNetDimI = 0
                if Plate.plateYmin % pixelSize == 0.:
                    minNetDimI = int(np.floor(Plate.plateYmin / pixelSize))
                else:
                    minNetDimI = int(np.floor(Plate.plateYmin / pixelSize)) + 1
                minNetDimI = minNetDimI - 2  # 나중에 상/하에 픽셀 1개씩 추가하기 때문에 지금 1개씩 제거
                netDimI = max(netDimI, minNetDimI)

                minNetDimJ = 0
                if Plate.plateXmin % pixelSize == 0.:
                    minNetDimJ = int(np.floor(Plate.plateXmin / pixelSize))
                else:
                    minNetDimJ = int(np.floor(Plate.plateXmin / pixelSize)) + 1
                minNetDimJ = minNetDimJ - 2  # 나중에 좌/우에 픽셀 1개씩 추가하기 때문에 지금 1개씩 제거
                netDimJ = max(netDimJ, minNetDimJ)
                # 발주 가능 최소 길이/폭 반영 위해 코드 추가(여기까지)

                netXmax = pixelSize * netDimJ
                netYmax = pixelSize * netDimI
                netArea = netXmax * netYmax
                netXmaxList = netXmaxList + [netXmax]
                netYmaxList = netYmaxList + [netYmax]
                netAreaList = netAreaList + [netArea]
                netDimJList = netDimJList + [netDimJ]
                netDimIList = netDimIList + [netDimI]

    else:
        pp = pList_Q2P_Record[0][0]
        pixelPlate = pp.PixelPlate
        idxI, idxJ = np.where(pixelPlate > 0)
        netDimI = max(idxI) - min(idxI) + 1
        netDimJ = max(idxJ) - min(idxJ) + 1

        # 발주 가능 최소 길이/폭 반영 위해 코드 추가(여기부터)
        minNetDimI = 0
        if Plate.plateYmin % pixelSize == 0.:
            minNetDimI = int(np.floor(Plate.plateYmin / pixelSize))
        else:
            minNetDimI = int(np.floor(Plate.plateYmin / pixelSize)) + 1
        minNetDimI = minNetDimI - 2  # 나중에 상/하에 픽셀 1개씩 추가하기 때문에 지금 1개씩 제거
        netDimI = max(netDimI, minNetDimI)

        minNetDimJ = 0
        if Plate.plateXmin % pixelSize == 0.:
            minNetDimJ = int(np.floor(Plate.plateXmin / pixelSize))
        else:
            minNetDimJ = int(np.floor(Plate.plateXmin / pixelSize)) + 1
        minNetDimJ = minNetDimJ - 2  # 나중에 좌/우에 픽셀 1개씩 추가하기 때문에 지금 1개씩 제거
        # 발주 가능 최소 길이/폭 반영 위해 코드 추가(여기까지)

        netXmax = pixelSize * netDimJ
        netYmax = pixelSize * netDimI
        netArea = netXmax * netYmax
        netXmaxList = [netXmax]
        netYmaxList = [netYmax]
        netAreaList = [netArea]
        netDimJList = [netDimJ]
        netDimIList = [netDimI]

    idxNetAreaMin = np.where(netAreaList == min(netAreaList))[0].tolist()
    idxNetAreaMin = np.where(np.array(netAreaList, dtype=float) == min(netAreaList))[0].tolist()

    if idxNetAreaMin == []: idxNetAreaMin = [0]  # 인덱스가 비어있는 경우가 있음.

    newPlateXmax = (1 + netDimJList[idxNetAreaMin[0]] + 1) * pixelSize  # 길이 방향 롤마진 고려하여 좌/우에 픽셀 하나씩 추가
    newPlateYmax = (1 + netDimIList[idxNetAreaMin[0]] + 1) * pixelSize  # 폭 방향 롤마진 고려하여 상/하에 픽셀 하나씩 추가

    new_pp = NewPlate(nnID, nList, newPlateXmax, newPlateYmax, 'NewPlate', pixelSize)

    pList = pList + [new_pp]
    P2QR(new_pp.ID, pList, QnTempID, qList, nnID, nList)  # , ShowProcess=True)

    ## 자재 사이즈 결정 2-2 단계: 부재의 배치 정보로 자재 사이즈 맞춤하기 - dot 이미지 용

    ## 자재 사이즈 결정 3단계: new_plateXmax ~ plateXmax(=21000.) 및 new_plateYmax ~ plateYmax(=4500.) 중 가장 저렴한 가격 찾기
    # to be implemented

    ## 자재 사이즈 결정 4단계: 가장 저렴한 가격에서 가장 넓은 것 찾기
    # to be implemented

    return new_pp.ID, pList


def Argmin_Loss(plateMargin, qq, minX, maxX, minY, maxY, ShowProcess=False):
    _, _, H_Plate, W_Plate = plateMargin.shape

    ### part pixel image formatting
    H_Part, W_Part, A_Part = qq.PixelPart.shape
    partMargin = np.zeros((A_Part, 1, H_Part, W_Part), dtype=int)
    for a in range(A_Part):
        partMargin[a, 0, :, :] = np.where(
            (np.array(qq.PixelPart[:, :, a], dtype=int) + np.array(qq.PixelMargin[:, :, a], dtype=int)) >= 1, 1, 0)

    ### Loss_Interference: 간섭 여부
    # lossOverlap = conv2d_ForLoop(plateMargin,partMargin) # for-loop 기반 연산.
    lossOverlap = conv2d_Im2Col(plateMargin, partMargin)  # im2col 기반 연산.
    lossOverlap = lossOverlap[0, :, :, :]  # 간섭이 없으면 0, 간섭이 있으면 1 이상
    lossOverlap = np.where(lossOverlap == 0, 0, np.inf)  # 간섭이 없으면 0, 간섭이 있으면 inf

    if np.min(lossOverlap) == np.inf:
        idxA = []
        idxH = []
        idxW = []
        allocationX = []
        allocationY = []
        allocationA = []
        minXTemp = minX
        maxXTemp = maxX
        minYTemp = minY
        maxYTemp = maxY
        return idxA, idxH, idxW, allocationA, allocationY, allocationX, minXTemp, maxXTemp, minYTemp, maxYTemp

    ### 자주 사용되는 값 미리 계산
    angleMinX_1D = np.zeros((A_Part, 1, 1), dtype='float')
    angleMaxX_1D = np.zeros((A_Part, 1, 1), dtype='float')
    angleMinY_1D = np.zeros((A_Part, 1, 1), dtype='float')
    angleMaxY_1D = np.zeros((A_Part, 1, 1), dtype='float')
    for a, angleDeg in enumerate(qq.AngleDegList):
        angleX, angleY = qq.DotRotating(qq.CDotUBurning, qq.CDotVBurning, angleDeg)
        angleMinX_1D[a, 0, 0] = np.min(angleX)
        angleMaxX_1D[a, 0, 0] = np.max(angleX)
        angleMinY_1D[a, 0, 0] = np.min(angleY)
        angleMaxY_1D[a, 0, 0] = np.max(angleY)

    angleCOGX_1D = np.zeros((A_Part, 1, 1), dtype='float')
    angleCOGY_1D = np.zeros((A_Part, 1, 1), dtype='float')
    for a, angleDeg in enumerate(qq.AngleDegList):
        angleCOGX, angleCOGY = qq.DotRotating(qq.UCOG, qq.VCOG, angleDeg)
        angleCOGX_1D[a, 0, 0] = angleCOGX
        angleCOGY_1D[a, 0, 0] = angleCOGY

    X_1D = np.ones((1, 1, W_Plate), dtype='float')
    for w in range(W_Plate): X_1D[0, 0, w] = pixelSize * (w + 0.5)  # Uhat = Umin + pixelSize*(idxJ+0.5)

    Y_1D = np.zeros((1, H_Plate, 1), dtype='float')
    for h in range(H_Plate): Y_1D[0, h, 0] = pixelSize * H_Plate - pixelSize * (
                h + 0.5)  # Vhat = Vmax - pixelSize*(idxI+0.5)

    A_1D = np.zeros((A_Part, 1, 1), dtype='float')
    for a, angleDeg in enumerate(qq.AngleDegList): A_1D[a, 0, 0] = angleDeg / 180 * np.pi

    ### Loss_Area_3D: 외접 사각형 면적
    minXold_3D = -lossOverlap + minX * np.ones((A_Part, 1, W_Plate), dtype='float')
    minXnew_3D = -lossOverlap + (X_1D + angleMinX_1D)
    minX_3D = np.minimum(minXold_3D, minXnew_3D)
    minX_3D = np.where(minX_3D == -np.inf, 0., minX_3D)

    maxXold_3D = lossOverlap + maxX * np.ones((A_Part, 1, W_Plate), dtype='float')
    maxXnew_3D = lossOverlap + (X_1D + angleMaxX_1D)
    maxX_3D = np.maximum(maxXold_3D, maxXnew_3D)
    maxX_3D = np.where(maxX_3D == np.inf, 0., maxX_3D)

    minYold_3D = -lossOverlap + minY * np.ones((A_Part, H_Plate, 1), dtype='float')
    minYnew_3D = -lossOverlap + (Y_1D + angleMinY_1D)
    minY_3D = np.minimum(minYold_3D, minYnew_3D)
    minY_3D = np.where(minY_3D == -np.inf, 0., minY_3D)

    maxYold_3D = lossOverlap + maxY * np.ones((A_Part, H_Plate, 1), dtype='float')
    maxYnew_3D = lossOverlap + (Y_1D + angleMaxY_1D)
    maxY_3D = np.maximum(maxYold_3D, maxYnew_3D)
    maxY_3D = np.where(maxY_3D == np.inf, 0., maxY_3D)

    lossArea_3D = lossOverlap + ((maxX_3D - minX_3D) * (maxY_3D - minY_3D))

    # 발주 가능 최소 길이/폭 적용(여기부터)
    # 원래 코드:
    # lossArea_3D = lossOverlap + ((maxX_3D-minX_3D)*(maxY_3D-minY_3D))
    # 수정 코드:
    # dX_3D = maxX_3D-minX_3D
    # dY_3D = maxY_3D-minY_3D
    #
    # plateXminWithoutRollMargin = Plate.plateXmin - 2*Plate.RollMarginX
    # plateYminWithoutRollMargin = Plate.plateYmin - 2*Plate.RollMarginY
    #
    # plateXminWithoutRollMargin_3D = lossOverlap + plateXminWithoutRollMargin*np.ones((1,1,W_Plate),dtype='float')
    # plateYminWithoutRollMargin_3D = lossOverlap + plateYminWithoutRollMargin*np.ones((1,H_Plate,1),dtype='float')
    #
    # plateXminWithoutRollMargin_3D = np.where(plateXminWithoutRollMargin==np.inf,0.,plateXminWithoutRollMargin)
    # plateYminWithoutRollMargin_3D = np.where(plateYminWithoutRollMargin==np.inf,0.,plateYminWithoutRollMargin)
    #
    # dX_3D = np.maximum(dX_3D,plateXminWithoutRollMargin_3D)
    # dY_3D = np.maximum(dY_3D,plateYminWithoutRollMargin_3D)
    #
    # lossArea_3D = lossOverlap + dX_3D*dY_3D
    # 발주 가능 최소 길이/폭 적용(여기까지)

    ### Loss_Iyy_3D: Iyy
    IyyTranslation = qq.Area * (X_1D + angleCOGX_1D) ** 2
    IyyRotation = qq.IuuCOG * np.sin(A_1D) ** 2 + qq.IvvCOG * np.cos(A_1D) ** 2 - 2 * qq.IuvCOG * np.cos(A_1D) * np.sin(
        A_1D)

    lossIyy_3D = lossOverlap + IyyTranslation + IyyRotation

    ### Loss_Ixx_3D: Ixx
    IxxTranslation = qq.Area * (Y_1D + angleCOGY_1D) ** 2
    IxxRotation = qq.IuuCOG * np.cos(A_1D) ** 2 + qq.IvvCOG * np.sin(A_1D) ** 2 + 2 * qq.IuvCOG * np.cos(A_1D) * np.sin(
        A_1D)

    lossIxx_3D = lossOverlap + IxxTranslation + IxxRotation

    ### Multilayered loss function: Ordering(Hyperparameter)
    loss1, threshold1 = lossArea_3D, 1.01  # 면적 최소화
    loss2, threshold2 = lossIyy_3D, 1.01  # 좌로 밀착
    loss3, threshold3 = lossIxx_3D, 1.01  # 아래로 밀착

    # 1. Select the best from Loss1
    loss1Min = np.min(loss1)
    idxA, idxH, idxW = np.where(loss1 <= loss1Min * threshold1)

    # 2. Select the best from Loss2 <-- Loss1
    newIdxA, newIdxH, newIdxW = [], [], []
    loss2Min = np.min(loss2[idxA, idxH, idxW])
    for ii in range(len(idxA)):
        loss2MinTemp = loss2[idxA[ii], idxH[ii], idxW[ii]]
        if loss2MinTemp <= loss2Min * threshold2:
            newIdxA = newIdxA + [idxA[ii]]
            newIdxH = newIdxH + [idxH[ii]]
            newIdxW = newIdxW + [idxW[ii]]
    idxA, idxH, idxW = copy.deepcopy(newIdxA), copy.deepcopy(newIdxH), copy.deepcopy(newIdxW)

    # 3. Select the best from Loss3 <-- Loss2 <-- Loss1
    newIdxA, newIdxH, newIdxW = [], [], []
    loss3Min = np.min(loss3[idxA, idxH, idxW])
    for ii in range(len(idxA)):
        loss3MinTemp = loss3[idxA[ii], idxH[ii], idxW[ii]]
        if loss3MinTemp <= loss3Min * threshold3:
            newIdxA = newIdxA + [idxA[ii]]
            newIdxH = newIdxH + [idxH[ii]]
            newIdxW = newIdxW + [idxW[ii]]
    idxA, idxH, idxW = copy.deepcopy(newIdxA), copy.deepcopy(newIdxH), copy.deepcopy(newIdxW)

    # 4. Select first one
    newIdxA, newIdxH, newIdxW = [idxA[0]], [idxH[0]], [idxW[0]]
    idxA, idxH, idxW = copy.deepcopy(newIdxA), copy.deepcopy(newIdxH), copy.deepcopy(newIdxW)

    minXTemp = X_1D[0, 0, idxW[0]] + angleMinX_1D[idxA[0], 0, 0]
    maxXTemp = X_1D[0, 0, idxW[0]] + angleMaxX_1D[idxA[0], 0, 0]

    minYTemp = Y_1D[0, idxH[0], 0] + angleMinY_1D[idxA[0], 0, 0]
    maxYTemp = Y_1D[0, idxH[0], 0] + angleMaxY_1D[idxA[0], 0, 0]

    allocationX = X_1D[0, 0, idxW[0]]
    allocationY = Y_1D[0, idxH[0], 0]
    allocationA = A_1D[idxA[0], 0, 0]

    # 발주 가능 최소 폭/길이 반영(여기부터)
    # 원래 코드: 없음
    # if (maxXTemp - minXTemp) < plateXminWithoutRollMargin:
    #    maxXTemp = minXTemp + plateXminWithoutRollMargin
    # if (maxYTemp - minYTemp) < plateYminWithoutRollMargin:
    #    maxYTemp = minYTemp + plateYminWithoutRollMargin
    # 발주 가능 최대 폭/길이 반영(여기까지)

    return idxA, idxH, idxW, allocationA, allocationY, allocationX, minXTemp, maxXTemp, minYTemp, maxYTemp




