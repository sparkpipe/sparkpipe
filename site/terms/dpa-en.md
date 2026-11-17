# SparkPipe Data Processing Agreement

Version of 13 November 2026

## Parties and scope

This Data Processing Agreement ("**DPA**") is made between **HYOR AI Tech Ltd.** (효르에이아이테크 유한회사), business registration number 641-87-03809 ("**we**"), and the Customer under the SparkPipe Customer Agreement (the "**Agreement**"). It forms part of the Agreement and applies whenever we process personal data on Customer's behalf in renting Customer the Fleets and the Extras, carrying its traffic to and from them, and providing our API ("**Customer personal data**"). Terms defined in the Agreement have the same meaning here.

## 1. Roles

1.1. For Customer personal data, Customer is the controller, or a processor acting for its own controller, and we are its processor. Under the Korean Personal Information Protection Act ("**PIPA**"), Customer entrusts the processing and we are the trustee.

1.2. For account, payment, booking and acceptance data, and the other records our privacy policy describes as ours, we are the controller. Our privacy policy applies to them, not this DPA.

1.3. This DPA is the contract that Article 28(3) of the EU General Data Protection Regulation ("**GDPR**"), the same article of the UK GDPR, and Article 26 of PIPA require.

## 2. Instructions and purpose

2.1. We process Customer personal data only on Customer's documented instructions. The Agreement, this DPA, the requests Customer sends to our API, its traffic to and from the Fleets it holds, the software and Functions it installs and runs, the Extras it uses and the settings it chooses are those instructions. Customer may give further reasonable instructions in writing.

2.2. We process Customer personal data only for the purpose in clause 2.4. We do not use it to train any model, sell it, combine it with other customers' data, or process it for any purpose of our own.

2.3. If we believe an instruction infringes data protection law, we tell Customer. If a law requires us to process Customer personal data otherwise, we tell Customer before we do, unless that law forbids it.

2.4. The details of the processing:

| Item | Details |
| --- | --- |
| Purpose and nature | Renting Customer the Fleets it holds and the Extras it uses: keeping its data on them, carrying its traffic to and from them and administering their machines, on which Customer runs the software it chooses |
| Data subjects | Whoever the data Customer keeps on, or sends to, the Fleets and Extras concerns, as Customer decides |
| Personal data | Whatever personal data Customer sends or stores; special categories only if Customer sends them, at its own responsibility |
| Duration | While Customer holds a Fleet or uses an Extra, until the deletion in clause 9 |
| Location | Only Fleets, Extras and servers we operate in Korea, including servers we rent there |

## 3. Confidentiality and access

3.1. Only the people who operate our Fleets and servers can access Customer personal data. Each of them is bound to confidentiality and accesses it only to operate and secure them, give support or meet a legal obligation.

## 4. Security

4.1. We take the technical and organisational measures in clause 12, and keep them at least at that level for as long as we process Customer personal data.

4.2. Customer is responsible for the security of its own systems and API keys, of the software it installs and runs on the Fleets and Extras, including keeping it updated, for the data it chooses to send, and for keeping its own copies, since we keep no backup of working state (Agreement clause 6.3).

## 5. Sub-processors

5.1. Customer authorises the sub-processor below. We use no other to process Customer personal data. A software module from the Catalogue runs only where Customer installs it, is not a sub-processor and sends Customer personal data nowhere Customer does not choose (Customer Agreement clause 7.8).

| Sub-processor | Task | Location |
| --- | --- | --- |
| KCInfra Inc. | Hosts the server we rent and operate, which carries Customer's traffic to and from the Fleets through an encrypted tunnel and stores none of it | Seoul, Korea |

5.2. Before we add or replace a sub-processor, we email Customer and update this DPA. Customer may object within 14 days on reasonable data protection grounds. If we cannot meet the objection, Customer may cancel the Bookings that the change would affect, and the HC spent on them is returned to its account.

5.3. We bind each sub-processor in writing to data protection obligations no less protective than this DPA, and remain responsible to Customer for its performance.

## 6. Transfers

6.1. We process Customer personal data only in Korea and transfer none of it abroad. What Customer, or software Customer installs or runs, sends or receives is not our transfer (Agreement clause 6.5).

6.2. The European Commission and the United Kingdom have each recognised Korea as providing an adequate level of protection, so personal data from the EU, the EEA and the UK may come to a Fleet without other transfer safeguards. If either decision stops applying, the European Commission's standard contractual clauses (controller-to-processor or processor-to-processor, as the case may be) or the UK addendum to them apply between the parties, and we sign them on request.

## 7. Data subjects' requests

7.1. Customer controls the data on the Fleets it holds and can, at any time, export the files in its Customer storage as Agreement clause 7.5 provides, or delete its working state. We help Customer, by appropriate technical and organisational measures and as far as possible, to answer data subjects who exercise their rights.

7.2. If a data subject contacts us about Customer personal data, we pass the request to Customer and do not answer it ourselves, unless Customer asks us to or the law requires it.

## 8. Personal data breaches

8.1. We tell Customer of a personal data breach affecting Customer personal data without undue delay, and at the latest 48 hours after we become aware of it. We say what happened, the categories and approximate amount of data concerned where we know them, the likely consequences, and what we have done and will do, and we add information as we learn it.

8.2. We help Customer meet its own duties to notify authorities and data subjects, and with any data protection impact assessment or prior consultation, taking into account the nature of the processing and the information available to us.

## 9. Deletion and return

9.1. Customer can export the files in its Customer storage at any time while it holds the Fleet, as Agreement clause 7.5 provides.

9.2. Working state is deleted when another customer's Booking of the Fleet starts, 24 hours after the end of Customer's last hour on it, or earlier when Customer asks (Agreement clause 6.2). We keep no copy outside the Fleet, except in an Extra Customer chooses, and make no backup. Data in an Extra is deleted when Customer stops using that Extra or the Agreement ends (Agreement clause 4.6).

## 10. Demonstrating compliance and audits

10.1. We make available the information needed to show that we meet this DPA: this DPA, our security page and privacy policy, and the source code we publish, which includes that of our inference engine and, where we publish it, that of the software that administers the Fleets and carries traffic to them. We answer Customer's reasonable written questions.

10.2. Customer, or an independent auditor bound to confidentiality, may audit our compliance once a year on 30 days' written notice, and also after a personal data breach affecting Customer personal data or when a supervisory authority requires it. Audits take place during business hours in Korea, at Customer's cost, and never include other customers' data or a Fleet another customer holds.

## 11. Liability, precedence, law and language

11.1. Liability under this DPA is subject to the limits in clause 8 of the Agreement, except where data protection law does not allow them.

11.2. If this DPA and the Agreement conflict on the processing of Customer personal data, this DPA prevails.

11.3. Korean law governs this DPA, and disputes are resolved as clause 12 of the Agreement provides. This does not limit any data subject's rights under the GDPR or the UK GDPR.

11.4. This DPA is published in Korean and English. The Korean text prevails.

## 12. Security measures

12.1. **Measures.** We take at least the measures below, and the further measures our security page describes. We may change those measures without amending this DPA, but never so that they protect Customer personal data less.

12.2. **Isolation.** A Fleet a customer holds runs only for that customer. Each customer's working state is isolated from other customers and stays on the Fleet that holds it, or in an Extra the customer chooses.

12.3. **Handover.** Before another customer's Booking of a Fleet starts, we delete the previous customer's accounts and working state on it and restore its machines to our base image.

12.4. **Encryption in transit.** All traffic to our website and our API uses HTTPS, and traffic between our servers runs through encrypted tunnels.

12.5. **Keys and credentials.** API keys are stored only as hashes and can be revoked at any time. A credential for another service that Customer sends to our API is held in memory only while the request that carries it runs, and is never stored, logged or shown.

12.6. **Access control.** Only the people who operate our Fleets and servers can reach our systems, each with a personal key, and we keep access records and review them after any incident.

12.7. **Physical security.** The Fleets are in premises we control in Korea, and our public server is in a data centre in Seoul.

12.8. **Transparency.** The source of our inference engine in the Catalogue is public, so Customer can read that engine's code, except that of separately licensed modules (Agreement clause 7.8).
