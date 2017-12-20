import csv

super_dict = {}
with open('./traffic.csv', 'rb') as csvfile:
  spamreader = csv.reader(csvfile, delimiter=',')
  for row in spamreader:
    # Parse Main Street & Near Streets
    tmp = row[2].split(' btwn ')
    if len(tmp) == 1:
      continue

    main = tmp[0][1:]
    tmp2 = tmp[1].split(' & ')
    if len(tmp2) == 1:
      tmp2 = tmp[1].split(' and ')
    if len(tmp2) == 1:
      continue
    near1 = tmp2[0]
    near2 = tmp2[1]
    
    #super_dict: near1, near2, direct, dow, type, 1:00, 2:00 .....
    if super_dict.get((main, near1, near2, int(row[4]), row[6])) is not None:
      for i in range(24):
        super_dict[(main, near1, near2, int(row[4]), row[6])][i] += int(row[7 + i])
        super_dict[(main, near1, near2, int(row[4]), row[6])][i] /= 2
    
    else:
      super_dict[(main, near1, near2, int(row[4]), row[6])] = [int(row[7]), int(row[8]), int(row[9]), int(row[10]), int(row[11]), int(row[12]), \
                                                               int(row[13]), int(row[14]), int(row[15]), int(row[16]), int(row[17]), int(row[18]), \
                                                               int(row[19]), int(row[20]), int(row[21]), int(row[22]), int(row[23]), int(row[24]), \
                                                               int(row[25]), int(row[26]), int(row[27]), int(row[28]), int(row[29]), int(row[30])]
search_dict = {}
for x in super_dict:
  new_list = super_dict[x]
  new_list.append(x[1])
  new_list.append(x[2])
  if search_dict.get((x[0],x[3],x[4])) is not None:
    search_dict[(x[0],x[3],x[4])].append(new_list)
  else:
    search_dict[(x[0],x[3],x[4])] = [new_list]

# Find traffic info for near1, near2
final_list = []
for keys in super_dict:
  roads_list = []
  main = keys[0]
  near1 = keys[1]
  near2 = keys[2]
  dow = keys[3]
  types = keys[4]
  
  if search_dict.get((near1,dow,types)) is not None:
    for x in search_dict[(near1, dow, types)]:
      if x[-1] == main or x[-2] == main:
        roads_list.append(x)
  
  if search_dict.get((near2,dow,types)) is not None:
    for x in search_dict[(near2, dow, types)]:
      if x[-1] == main or x[-2] == main:
        roads_list.append(x) 
  
  if search_dict.get((main,dow,types)) is not None:
    for x in search_dict[(main,dow,types)]:
      tup = {x[-1], x[-2]}
      if (near1 in tup and near2 not in tup) or (near1 not in tup and near2 in tup):
        roads_list.append(x)

  if len(roads_list) == 0:
    continue
  
  nearby_list = [0, 0, 0, 0, 0, 0, \
                 0, 0, 0, 0, 0, 0, \
                 0, 0, 0, 0, 0, 0, \
                 0, 0, 0, 0, 0, 0]

  for x in roads_list:
    for i in range(24):
      nearby_list[i] += x[i]

  for i in range(24):
    cur_traf = super_dict[keys][i]
    next = i + 1

    if next <= 23:
      next_traf = super_dict[keys][next]
    else:
      next_traf = super_dict[keys][24 - next]  
    
    near_traf = nearby_list[i]
    
    last = i - 1
    if (last) >= 0:
      gradient = super_dict[keys][i] - super_dict[keys][last]
    else:
      gradient = super_dict[keys][i] - super_dict[keys][last + 24]

    if next_traf >= 1000:
      pred = 2
    elif next_traf >= 100:
      pred = 1
    else:
      pred = 0
    #final_list.append([main, pred, cur_traf, near_traf, i])
    final_list.append([pred, gradient, cur_traf, near_traf, i])

with open('traffic_grad.csv', 'wb') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    #spamwriter.writerow(["Street_Name","Traffic_Nexttime-Pred", "Current_Traffic","Near_Traffic","Time"])
    for x in final_list:
      spamwriter.writerow(x)
