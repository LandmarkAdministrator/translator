# Opus-MT versus NLLB-200, head to head

**2026-09-23.** Same 114 sentences from the sermon transcript, same machine
(RTX 3060), same run. NLLB-1.3B is what production uses today; Opus-MT is
three separate English-to-X models, which is how it is meant to be deployed.
Full data: `benchmarks/REPORT-combined-2026-09-23.txt`.

## What changes on the machine

| | NLLB-1.3B | Opus-MT | change |
|---|---|---|---|
| licence | CC-BY-NC-4.0, no commercial use | CC-BY-4.0, attribution only | the point of the exercise |
| per translation, RTX 3060 | 0.161 s | 0.029 s | 5.5x faster |
| per translation, Radeon 890M | 0.806 s | 0.064 s | 12x faster |
| per translation, CPU | 2.667 s | 0.215 s | 12x faster, and usable |
| GPU memory | 2.6 GB | 441 MB | a sixth |
| three languages at once | one model, one lock, no gain from copies | three models, 2.05x on CPU | languages stop competing |
| adding a language | a FLORES code | find an `opus-mt-en-XX` model | not all pairs exist, and quality varies per pair |

The headline is not the speed, it is that **translation stops needing a GPU**.
At 0.215 s on CPU, three languages fit inside the gap between sentences.

## Quality, measured over all 114 sentences

| Signal | Spanish | Creole | Russian |
|---|---|---|---|
| identical output | 28 of 114 | 2 of 114 | 13 of 114 |
| kept "tonight" — NLLB / Opus | 13 / 13 | **13 / 1** | 12 / 13 |
| used the sermon's word "spirit" — NLLB / Opus | 19 / 19 | **17 / 5** | 18 / 19 |
| dropped content, under 60% of source length | 2 / 2 | 1 / 1 | 6 / 3 |
| English left untranslated | 0 / 0 | 0 / 0 | 0 / 0 |
| loops, from the main report | 0.3% / 0% | 0.3% / 0% | 0.3% / 0% |

Spanish and Russian are a draw. **Creole is not**, and the two Creole rows
above are the whole argument: Opus drops the time reference in 12 of 13
sentences that carry one, and it replaces the sermon's key word.

## Creole

This is where they differ. Read the last three especially.

> **EN** I hope it's well with your soul tonight.
>
> **NLLB** Mwen espere ke se byen ak nanm ou aswè a.
>
> **Opus** Mwen espere ke li byen ak nanm ou.

> **EN** Now, I'm not sure tonight, I believe that I'll be preaching.
>
> **NLLB** Koulye a, mwen pa sèten aswè a, mwen kwè ke mwen pral preche.
>
> **Opus** Kounye a, mwen pa sèten de sa, mwen kwè m ap preche.

> **EN** Now, I want to draw your attention tonight to some things that are in the passage, and hopefully that you'll be able to take some things with you about this.
>
> **NLLB** Koulye a, mwen vle atire atansyon ou aswè a nan kèk bagay ki nan pasaj la, e espere ke ou pral kapab pran kèk bagay avèk ou sou sa.
>
> **Opus** Koulye a, mwen vle fè atansyon ak kèk bagay ki nan pasaj sa a, e, ak espwa ke w ap kapab pran kèk bagay avè w sou sa.

> **EN** But I want to encourage you tonight, again, have an excellent spirit.
>
> **NLLB** Men, mwen vle ankouraje nou aswè a, ankò, gen yon lespri ekselan.
>
> **Opus** Men, mwen vle ankouraje w ankò, gen yon bèl etadespri.

> **EN** I want you to do this tonight as I pray and ask the Lord to help me.
>
> **NLLB** Mwen vle ou fè sa aswè a pandan m ap priye epi mande Senyè a pou l ede m.
>
> **Opus** Mwen vle w fè sa toutpandan m ap priye Seyè a pou m mande l ede m.

> **EN** Thank you, Brother Oar.
>
> **NLLB** Mèsi, frè Rem.
>
> **Opus** Mèsi, frè O.

> **EN** Take your Bibles and go to Daniel chapter number 6.
>
> **NLLB** Pran Bib nou yo e gade Danyèl chapit 6.
>
> **Opus** Pran Bib ou epi al nan Dànyèl chapit 6.

> **EN** Daniel chapter number 6 in the Old Testament.
>
> **NLLB** Danyèl chapit 6 nan Ansyen Testaman an.
>
> **Opus** Dànyèl chapit 6 nan Ansyen Testaman an.

> **EN** I appreciate the good music and I said it before, do pray for a lot of our folks who are sick, not sure, a lot of folks out of town.
>
> **NLLB** Mwen apresye bon mizik la e mwen te di l 'devan, priye pou yon anpil nan moun nou yo ki malad, pa sèten, yon anpil nan moun ki soti nan vil la.
>
> **Opus** Mwen apresye bon mizik yo te konn pase anvan m te di sa, mwen priye pou anpil nan moun ki nan pèp la ki malad, ki pa fin sèten, ki soti nan yon pakèt moun nan vil la.

> **EN** And so if you do see somebody, either let the church office know, let us know.
>
> **NLLB** Se konsa, si ou wè yon moun, swa fè biwo legliz la konnen, fè nou konnen.
>
> **Opus** Si w wè yon moun, oswa si w kite biwo legliz la konnen, annou fè nou konnen.

> **EN** If there's somebody that you reached out to and you found out they were sick, or maybe you reached out and they're out of town and they're doing well, they'll be glad that you reached out to them.
>
> **NLLB** Si gen yon moun ou te kontakte epi ou te jwenn ke yo te malad, oswa petèt ou te kontakte yo epi yo deyò vil la epi yo ap fè byen, yo pral kontan ke ou te rive jwenn yo.
>
> **Opus** Si gen yon moun ou te rive jwenn yo malad, petèt ou te rive soti nan vil la, epi yo ap byen fè, yo ap kontan dèske w te rive jwenn yo.

> **EN** There's a lot of folks that we think about sometimes, oh, somebody else is reaching out to them.
>
> **NLLB** Gen anpil moun ke nou panse sou pafwa, oh, yon lòt moun ap rive jwenn yo.
>
> **Opus** Gen anpil moun ki gen repitasyon nou panse pafwa, o, yon lòt moun ap chèche pou yo.

## Spanish

Representative of the whole set: different wording, same meaning.

> **EN** Thank you, Brother Oar.
>
> **NLLB** Gracias, hermano Remador.
>
> **Opus** Gracias, hermano Oar.

> **EN** I hope it's well with your soul tonight.
>
> **NLLB** Espero que esté bien con su alma esta noche.
>
> **Opus** Espero que te vaya bien esta noche.

> **EN** Take your Bibles and go to Daniel chapter number 6.
>
> **NLLB** Tomen sus Biblias y vayan al capítulo número 6 de Daniel.
>
> **Opus** Tomen sus Biblias e vayan al capítulo 6 de Daniel.

> **EN** I appreciate the good music and I said it before, do pray for a lot of our folks who are sick, not sure, a lot of folks out of town.
>
> **NLLB** Aprecio la buena música y lo dije antes, rezar por muchos de nuestros amigos que están enfermos, no estoy seguro, muchos fuera de la ciudad.
>
> **Opus** Aprecio la buena música y lo dije antes, reza por muchos de nuestros padres que están enfermos, no estoy seguro, mucha gente fuera de la ciudad.

> **EN** And so if you do see somebody, either let the church office know, let us know.
>
> **NLLB** Y si ves a alguien, avísalo a la oficina de la iglesia, avísalo a nosotros.
>
> **Opus** Y si ves a alguien, o avisa a la oficina de la iglesia, háznoslo saber.

> **EN** If there's somebody that you reached out to and you found out they were sick, or maybe you reached out and they're out of town and they're doing well, they'll be glad that you reached out to them.
>
> **NLLB** Si hay alguien a quien has contactado y has descubierto que está enfermo, o tal vez lo has contactado y está fuera de la ciudad y lo está haciendo bien, estarán contentos de que lo hayas contactado.
>
> **Opus** Si hay alguien a quien has contactado y te has enterado de que estaba enfermo, o tal vez has contactado y están fuera de la ciudad y lo están haciendo bien, se alegrarán de que hayas contactado con ellos.

> **EN** There's a lot of folks that we think about sometimes, oh, somebody else is reaching out to them.
>
> **NLLB** Hay mucha gente en la que pensamos a veces, oh, alguien más está llegando a ellos.
>
> **Opus** Hay mucha gente en la que pensamos a veces, oh, alguien más se está acercando a ellos.

> **EN** But if you notice that God brought it to your attention, and it's good to reach out to them and encourage them.
>
> **NLLB** Pero si usted nota que Dios lo trajo a su atención, y es bueno llegar a ellos y animarlos.
>
> **Opus** Pero si usted nota que Dios lo trajo a su atención, y es bueno que se acerca a ellos y los anima.

> **EN** The book of Daniel here is, I mean, Daniel at the beginning of the book, and we've used this passage of Scripture in the early part, but we realize that he determined not to defile himself.
>
> **NLLB** El libro de Daniel aquí es, quiero decir, Daniel al principio del libro, y hemos usado este pasaje de la Escritura en la primera parte, pero nos damos cuenta de que él decidió no contaminarse.
>
> **Opus** El libro de Daniel aquí es, quiero decir, Daniel al principio del libro, y hemos usado este pasaje de la Escritura en la parte temprana, pero nos damos cuenta de que él decidió no contaminarse a sí mismo.

> **EN** So he began in this thing with good decisions, and it's continued to the place where he comes to chapter six, and we're going to see him with an excellent spirit.
>
> **NLLB** Así que comenzó en esta cosa con buenas decisiones, y se ha continuado hasta el lugar donde llega al capítulo seis, y vamos a verlo con un espíritu excelente.
>
> **Opus** Así que comenzó en esta cosa con buenas decisiones, y ha continuado hasta el lugar donde viene al capítulo seis, y vamos a verlo con un espíritu excelente.

## Russian

Also a draw; the differences are register, not meaning.

> **EN** Thank you, Brother Oar.
>
> **NLLB** Спасибо, брат Весло.
>
> **Opus** Спасибо, брат Оар.

> **EN** I hope it's well with your soul tonight.
>
> **NLLB** Надеюсь, сегодня твоя душа в порядке.
>
> **Opus** Надеюсь, сегодня твоя душа будет в порядке.

> **EN** Take your Bibles and go to Daniel chapter number 6.
>
> **NLLB** Возьмите свои Библии и откройте Даниил, глава 6.
>
> **Opus** Возьми свои Библии и отправься в 6 -ю главу Даниила.

> **EN** Daniel chapter number 6 in the Old Testament.
>
> **NLLB** Даниил, глава 6 Ветхого Завета.
>
> **Opus** Дэниел, глава номер 6, в Ветхом Завете.

> **EN** I appreciate the good music and I said it before, do pray for a lot of our folks who are sick, not sure, a lot of folks out of town.
>
> **NLLB** Я ценю хорошую музыку, и я уже говорил, что молюсь за больных, не уверенных, за тех, кто не в городе.
>
> **Opus** Я ценю хорошую музыку и я сказал это раньше, молитесь за многих наших родителей, которые больны, не уверен, много людей из города.

> **EN** And so if you do see somebody, either let the church office know, let us know.
>
> **NLLB** И если вы кого-нибудь увидите, сообщите в церковный офис, сообщите нам.
>
> **Opus** И если вы увидите кого-нибудь, либо сообщите в церковный офис, либо дайте нам знать.

> **EN** If there's somebody that you reached out to and you found out they were sick, or maybe you reached out and they're out of town and they're doing well, they'll be glad that you reached out to them.
>
> **NLLB** Если вы связались с кем-то, и вы узнали, что он болен, или, может быть, вы связались с ним, и он не в городе, и он хорошо себя чувствует, он будет рад, что вы связались с ним.
>
> **Opus** Если кто-то, к кому ты связался и узнал, что они больны, или, возможно, ты связался с ним, и они уехали из города, и они хорошо справляются, они будут рады, что ты связался с ними.

> **EN** There's a lot of folks that we think about sometimes, oh, somebody else is reaching out to them.
>
> **NLLB** Есть много людей, о которых мы иногда думаем, о, кто-то другой пытается связаться с ними.
>
> **Opus** Мы иногда думаем о многих людях, о которых кто-то другой тянется к ним.

> **EN** But if you notice that God brought it to your attention, and it's good to reach out to them and encourage them.
>
> **NLLB** Но если вы заметили, что Бог обратил на это ваше внимание, и это хорошо, чтобы обратиться к ним и поощрить их.
>
> **Opus** Но если вы заметили, что Бог обратил на это внимание, и это хорошо, что вы общаетесь с ними и поддерживаете их.

> **EN** As a church, we ought to be doing that.
>
> **NLLB** Как церковь, мы должны делать это.
>
> **Opus** Как церковь, мы должны это делать.


## Analysis

**Spanish and Russian: switch freely.** Across 114 sentences the two models
are interchangeable in meaning. Both keep every number, neither loops,
neither leaves English in the output, and both hold the sermon's vocabulary.
The differences are the kind two human translators would produce. Opus is
slightly better in places (`la guarida del león` against NLLB's non-standard
`la cova de los leones`) and slightly worse in others (`folks` became
`padres`, parents, once in each language; one missing subjunctive). Nothing
a listener would notice as an error.

**Creole: they are not interchangeable, and NLLB is better.** Three separate
measurements point the same way.

1. *Time references disappear.* Of 13 sentences containing "tonight", NLLB
   carried it into Creole 13 times, Opus once. A preacher says "tonight"
   constantly, and each loss shifts a sentence from this service to no
   particular time.
2. *The sermon's own word drifts.* The passage is about Daniel's "excellent
   spirit". NLLB wrote `lespri` in 17 of 19 sentences about it; Opus in 5,
   substituting `mantalite`, `etadespri` and once `espwa`, hope. A listener
   following the preacher's repeated phrase loses the thread.
3. *Clauses go missing or invert.* "That you would take something home
   tonight that helps you in your Christian life" became "That can help you
   in your Christian life" — the request is gone. "I pray God tonight that
   you would use me" became a prayer that God would help *you* understand
   something. "With a portion of the king's meat", a fragment, gained an
   invented subject: "He had a portion".

That pattern is consistent with what the two models are. NLLB-200 was trained
with a deliberate low-resource effort behind Haitian Creole. `opus-mt-en-ht`
is a small older model built largely from religious and relief-era text; it
is fluent on short simple sentences and compresses long spoken ones, which is
exactly the wrong shape for a preached sermon.

**So: keep NLLB for Creole; Opus is ready for Spanish and Russian today.**
Because each language already runs its own pipeline, they do not have to use
the same model. Mixing is a configuration change, not an architecture change.

That leaves the licence exactly where it was for one language of three. For
the church's own use nothing blocks NLLB. For a product, the options are to
accept Opus's weaker Creole, to fence it (a short-phrase dictionary and a
rule that catches dropped time words would remove the worst of it), or to
fine-tune a permissive model on Creole sermon text — the archive holds
thousands of aligned English and Creole sentences, though they were produced
by NLLB, which is its own licensing question.

**What no measurement here can settle:** whether Opus's Creole is *good
enough* for the congregation even with those faults. The samples above are
the sheet to hand a Creole speaker. My reading says NLLB is clearly more
faithful; a native listener may weigh fluency differently.

---

## Bigger sample: the whole 56-minute test suite (2026-09-23)

The comparison above used one 10-minute sermon, 114 sentences. Repeated over
all five tracks of `suite_v4`: sermon, JFK's inaugural, a Stanford
commencement address, congregational singing, and read speech.

**396 sentences, 7,144 words, 1,188 translations per model**, three models,
on the RTX 3060. MADLAD-400 3B is included since it is the other permissive
candidate.

| over 348 sentences of cased speech ×3 languages | NLLB | Opus-MT | MADLAD-3B |
|---|---|---|---|
| seconds per translation | 0.190 | **0.040** | 0.490 |
| GPU memory | 2.6 GB | **443 MB** | 5.7 GB |
| loops | 0.8% | 1.6% | 1.9% |
| **numbers kept** | **94%** | **76%** | 92% |
| **numbers kept, Creole only** | **100%** | **59%** | 94% |
| content dropped | 1.8% | 2.1% | 0.6% |
| length vs source | 1.00 | 1.02 | 1.06 |

**One track was excluded as an artefact.** The LibriSpeech reference is in
ALL CAPITALS, and Opus collapses on it — 67% loop rate in Creole, output like
"YON MOUN KI TE YON MOUN KI TE…". Lowercase the same sentences and it
translates them properly, so the trigger is the capitals, not the content.
Our ASR emits cased text, so this will not occur in production. It is still a
fair warning that Opus breaks on input outside its training distribution
where NLLB simply coped.

### The finding that matters: Opus drops numbers

With 51 numbers in the corpus instead of 2, the signal finally has weight,
and it is the clearest difference between the two models.

> **EN** Let's stand together, please, and turn in your hymn books again to 413.
> **NLLB** Ann kanpe ansanm, tanpri, epi vire liv imn ou ankò nan 413.
> **Opus** An n kanpe ansanm, tanpri, vire nan liv ou yo ankò pou **4 1513**.

> **EN** I had a scan at 7:30 in the morning, and it clearly showed a tumor on my pancreas.
> **NLLB** Mwen te fè yon tès done nan 7:30 maten an, e li te montre yon timè sou pankreyas mwen.
> **Opus** Mwen te gen yon pwoblèm nan maten, e li te montre m yon kansè sou **janm** mwen.
> *(the time is gone, and the tumour moved from the pancreas to the leg)*

> **EN** It was the mid-1970s, and I was your age.
> **NLLB** Se te nan mitan ane 1970 yo, e mwen te gen laj ou.
> **Opus** Se te **mitan mitan a**, e m te gen laj.

In Creole, Opus carried through 59% of the numbers; NLLB carried 100%. For a
service built on "turn to chapter 6, verse 10" and "hymn number 413", that is
the difference that would be noticed first.

### What the bigger sample did *not* change

The sermon-track conclusions hold: Spanish and Russian are a draw, Creole
favours NLLB, and the speed and memory gap is unchanged — Opus is still about
five times faster and needs a sixth of the memory.

### Revised recommendation

Unchanged for Spanish and Russian: **Opus is ready today**. For Creole the
case against it is now stronger, and specific: it is not only the dropped time
references and the drifting vocabulary, it is the numbers.

The number problem is worth noting because it is *already on the to-do list
from the other end*: `TODO.md` item 8 records that the Creole and Russian
voices cannot pronounce digits. One stage fixes both — protect numbers across
translation, then spell them out before the voice. Build that and Opus loses
its worst failure mode in the process.

---

## The number guard (2026-09-23)

Since dropped numbers were the clearest fault, it is worth fixing rather than
tolerating. `src/pipeline/number_guard.py`, 35 tests in
`tests/test_number_guard.py`.

### Two designs measured and rejected

- **Placeholders everywhere.** Replace each number with a marker, translate,
  put it back. Markers survive Spanish (100%) and Russian (93%) but **none
  survives Creole** — `#0#`, `NUM0`, `X0`, `@0`, `[0]`, `§0` all scored 0%.
- **Spelling numbers out in English first.** Correct for small numbers
  ("chapter six" works) and actively harmful above about ten: "when I was 17"
  became *sèt an* (seven), "at 30" became *nan tren* (in the train), "turned
  30" lost the number entirely.

### What it does instead

Leave the digits in place, check the result, repair only a real loss.

1. Translate normally.
2. Check each source number survived **in any fair form** — the same digits,
   a shortened year (1970 as "los años 70", "70-х"), the parts of a time
   (7:30 as "7 è 30"), or spelled out in the target, reusing the Creole and
   Russian number words already written for the voices.
3. If something genuinely vanished: for Spanish and Russian retranslate with
   placeholders, which work there; otherwise translate the words around each
   number and set the digits back in place, without letting a fragment end
   the sentence early.

**A first version was too strict and made things worse.** It counted "a
mediados de los años 70" as a lost number and rewrote that good sentence into
"Era la mitad... 1970 s". Recognising legitimate forms is what separates a
guard from a vandal.

### Results, same suite, same machine

| | numbers kept | Creole | seconds | sentences changed |
|---|---|---|---|---|
| NLLB, guard off | 100% | 100% | 0.190 | — |
| NLLB, guard on | 100% | 100% | 0.190 | **0** |
| **Opus, guard off** | 87% | **67%** | 0.040 | — |
| **Opus, guard on** | **100%** | **100%** | 0.041 | 5 of 1,044 |
| MADLAD, guard off | 98% | 93% | 0.490 | — |
| MADLAD, guard on | 100% | 100% | 0.491 | 1 of 1,044 |

It costs 2.5% of translation time, touches 0.5% of sentences, and never fired
on NLLB at all — which is the evidence that it is not rewriting healthy
output.

The repairs that matter:

> **EN** Let's stand together, please, and turn in your hymn books again to 413.
> **unguarded** …vire nan liv ou yo ankò pou **4 1513**.
> **guarded** …vire tèt tounen liv ou yo ankò **413**.

> **EN** I had a scan at 7:30 in the morning…
> **unguarded** Mwen te gen yon pwoblèm **nan maten**…
> **guarded** Mwen te gen yon pwoblèm nan men m **7:30** nan maten…

Honest about the cost: two of the five repairs read a little more clumsily
than the originals, because translating around a figure loses the inflection
a language would put on it. The numbers are right and the sentences are
understandable, which for chapter, verse and hymn numbers is the trade worth
making.

### Where it is not yet

The guard is used by the benchmark only. Putting it into production means one
call in `src/pipeline/translation.py`, and it pairs with the voice-side
spelling in `src/pipeline/number_words.py` that already exists but is likewise
not wired in: the guard keeps digits in the text people read, the verbalizer
turns them into words for voices that cannot say digits.
