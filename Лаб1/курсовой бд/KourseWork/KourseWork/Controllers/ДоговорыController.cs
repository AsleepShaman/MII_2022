using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Data;
using System.Data.Entity;
using System.Diagnostics;
using System.Linq;
using System.Net;
using System.Web;
using System.Web.Mvc;
using KourseWork.Models;
using System.Data.SqlClient;

namespace KoursWork.Controllers
{
    public class ДоговорыController : Controller
    {
        private AudioStudioDBEntities2 db = new AudioStudioDBEntities2();

        // GET: Договоры

        public ActionResult Index()
        {
            var договоры = db.Договоры.Include(д => д.Исполнитель);
            return View(договоры.ToList());
        }

        [HttpPost]
        public ActionResult Index(string DogovorNumber, string period_1, string period_2)
        {
            var договоры = db.Договоры.Include(д => д.Исполнитель);
            try
            {
                if (!string.IsNullOrEmpty(DogovorNumber))
                {
                    using (AudioStudioDBEntities2 db = new AudioStudioDBEntities2())
                    {
                        System.Data.SqlClient.SqlParameter parameter = new System.Data.SqlClient.SqlParameter("@Dog_count", DogovorNumber);
                        db.Database.ExecuteSqlCommand("Dogovor @Dog_count", parameter);
                    }
                }
            }
            catch
            {
                ViewBag.Data = "Пустой договор!";
                return RedirectToAction("Error");
            }

               if(!string.IsNullOrEmpty(period_1) && (!string.IsNullOrEmpty(period_2)))
                {

                decimal? rat = db.PeriodSell(DateTime.Parse(period_1), DateTime.Parse(period_2)).First().Salary;
                ViewBag.Result = "Выручка за данный период: " + rat + ". ";
                return View(договоры.ToList());

            }

            return View(договоры.ToList());
        }
        public ActionResult Error()
        {
            return View();
        }

        // GET: Договоры/Details/5
        public ActionResult Details(string id)
        {
            if (id == null)
            {
                return RedirectToAction("Error404", "Error");
            }
            Договоры договоры = db.Договоры.Find(id);
            if (договоры == null)
            {
                return RedirectToAction("Error404", "Error");
            }
            return View(договоры);
        }

        // GET: Договоры/Create
        public ActionResult Create()
        {
            ViewBag.Паспорт_заказчика = new SelectList(db.Исполнитель, "Паспорт_Исполнителя", "ФИО");
            return View();
        }

        // POST: Договоры/Create
        // Чтобы защититься от атак чрезмерной передачи данных, включите определенные свойства, для которых следует установить привязку. Дополнительные 

        [HttpPost]
        [ValidateAntiForgeryToken]
        public ActionResult Create([Bind(Include = "Номер_договора,Стоимость,Дата_заключения,Кол_во_услуг,Паспорт_заказчика")] Договоры договоры)
        {
            if (ModelState.IsValid)
            {
                try
                {
                    db.Договоры.Add(договоры);
                    db.SaveChanges();
                    return RedirectToAction("Index");
                }
                catch
                {
                    ViewBag.Data = "Ошибка создания пользователя.";
                    return RedirectToAction("Error");
                }

            }

            ViewBag.Паспорт_заказчика = new SelectList(db.Исполнитель, "Паспорт_Исполнителя", "ФИО", договоры.Паспорт_заказчика);
            return View(договоры);
        }

        // GET: Договоры/Edit/5
        public ActionResult Edit(string id)
        {
            if (id == null)
            {
                return RedirectToAction("Error404", "Error");
            }
            Договоры договоры = db.Договоры.Find(id);
            if (договоры == null)
            {
                return RedirectToAction("Error404", "Error");
            }
            ViewBag.Паспорт_заказчика = new SelectList(db.Исполнитель, "Паспорт_Исполнителя", "ФИО", договоры.Паспорт_заказчика);
            return View(договоры);
        }

        // POST: Договоры/Edit/5
        // Чтобы защититься от атак чрезмерной передачи данных, включите определенные свойства, для которых следует установить привязку. Дополнительные 

        [HttpPost]
        [ValidateAntiForgeryToken]
        public ActionResult Edit([Bind(Include = "Номер_договора,Стоимость,Дата_заключения,Кол_во_услуг,Паспорт_заказчика")] Договоры договоры)
        {
            if (ModelState.IsValid)
            {
               try{ db.Entry(договоры).State = EntityState.Modified;
                db.SaveChanges();
                return RedirectToAction("Index");
                }
                catch
                {
                    ViewBag.Data = "Ошибка редактирования!";
                    return RedirectToAction("Error");
                }
            }
            ViewBag.Паспорт_заказчика = new SelectList(db.Исполнитель, "Паспорт_Исполнителя", "ФИО", договоры.Паспорт_заказчика);
            return View(договоры);
        }

        // GET: Договоры/Delete/5
        public ActionResult Delete(string id)
        {
            if (id == null)
            {
                return RedirectToAction("Error404", "Error");
            }
            Договоры договоры = db.Договоры.Find(id);
            if (договоры == null)
            {
                return RedirectToAction("Error404", "Error");
            }
            return View(договоры);
        }

        // POST: Договоры/Delete/5
        [HttpPost, ActionName("Delete")]
        [ValidateAntiForgeryToken]
        public ActionResult DeleteConfirmed(string id)
        {
            try{Договоры договоры = db.Договоры.Find(id);
            db.Договоры.Remove(договоры);
            db.SaveChanges();
            return RedirectToAction("Index");
            }
            catch
            {
                ViewBag.Data = "Ошибка удаления! Возможно, эти данные где-то есть...";
                return RedirectToAction("Error");
            }
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                db.Dispose();
            }
            base.Dispose(disposing);
        }
    }
}
